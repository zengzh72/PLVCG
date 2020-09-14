#encoding=utf-8
# modified from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bart.py
import torch
import torch.nn as nn
import torch.nn.functional as F


import json
import math
import random
import numpy as np
from typing import NamedTuple, Optional, Iterable, Tuple


from transformers import BartForConditionalGeneration, BartModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_bart import BartEncoder, invert_mask, _prepare_bart_decoder_inputs, Seq2SeqModelOutput
from transformers.modeling_bart import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding, EncoderLayer, LayerNorm, BartClassificationHead
from transformers.configuration_bart import BartConfig
from transformers.generation_utils import top_k_top_p_filtering

class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = 30007 # Size of Vocabulary
    dim: int = 1024 # Dimension of Hidden Layer in Transformer Encoder
    encoder_layers: int = 6 # Numher of Encoder Layers
    decoder_layers: int = 6
    n_heads: int = 8 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_n_clips: int = 5        # Maximum video clips for each comment
    max_comment_len: int = 20   # Maximun words for each comment 
    max_context_len: int = 128  # Maximum words for context comments
    max_len : int = 154
    video_type_num: int = 19
    clf_weight : float = 50.0
    
    @classmethod
    def load_from_json(cls, file):
        return cls(**json.load(open(file, "r")))
 
 
class  BartEncoderWithVisual(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__(config, embed_tokens)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.visual = None

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
        self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_tuple=False, visual=None
    ):
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self.visual is not None:
            visual = self.visual
         
        inputs_embeds = torch.cat([visual,inputs_embeds], dim=1) 
        
        visual_zeros = torch.zeros([visual.size()[0],visual.size()[1]], dtype=input_ids.dtype).to(torch.device("cuda"))
        embed_pos = self.embed_positions(torch.cat([visual_zeros,input_ids], dim=1))
        
        
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if return_tuple:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)



class BartModelWithVisual(BartModel):
    
    def __init__(self, config: BartConfig, type='pretrain'):
        super().__init__(config)
        self.encoder = BartEncoderWithVisual(config, self.shared)
        self.type=type
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_tuple=None,
        **kwargs,
    ):

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_tuple = return_tuple if return_tuple is not None else self.config.use_return_tuple

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None
        
        causal_mask[0,1] = 0

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_tuple=return_tuple,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_tuple=False
        elif not return_tuple and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_past_key_values=decoder_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_tuple=return_tuple,
        )

        if return_tuple:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
class BartForConditionalGenerationWithVisual(BartForConditionalGeneration):
    
    def __init__(self, config: BartConfig, type):
        super().__init__(config)
        self.model = BartModelWithVisual(config, type)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
    


class MyPLVCG(BartForConditionalGenerationWithVisual):
    def __init__(self, model_cfg, clf_loss_weight=None, type="pretrain"):
        config = BartConfig(
                    vocab_size = model_cfg.vocab_size,
                    d_model = model_cfg.dim,
                    encoder_ffn_dim = model_cfg.dim_ff,
                    encoder_layers = model_cfg.encoder_layers,
                    encoder_attention_heads = model_cfg.n_heads,
                    decoder_ffn_dim = model_cfg.dim_ff,
                    decoder_layers = model_cfg.decoder_layers,
                    decoder_attention_heads = model_cfg.n_heads,
                    attention_dropout = model_cfg.p_drop_attn,
                    dropout = model_cfg.p_drop_hidden,
                    max_position_embeddings = model_cfg.max_context_len + model_cfg.max_n_clips + 1,
                    num_labels = 2,
                    pad_token_id = 0,
                    bos_token_id = 1,
                    eos_token_id = 2,
                    unk_token_id = 3,
                    num_beams = 1,
                    is_encoder_decoder = True
                )
        self.config = config
        self.type = type
        super().__init__(config, type)
        self.model_cfg = model_cfg 
        self.encoder = self.model.encoder
        
        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, model_cfg.video_type_num, 0,
        )
        if clf_loss_weight is not None:
            self.clf_loss_weight = torch.Tensor(clf_loss_weight).to(torch.device("cuda"))
    
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, video_types=None,
            decoder_cached_states=None,
            use_cache=False, is_training=False):


        decoder_attention_mask_input = decoder_attention_mask
            
        outputs = self.model(
            input_ids,
            attention_mask=self.attach_visual_for_mask(attention_mask),
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask_input,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        
        lm_logits = F.linear(outputs[0][:,1:-1], self.model.shared.weight, bias=self.final_logits_bias)     
        decoder_input_ids_gt = decoder_input_ids[:,2:]
        decoder_attention_mask_gt = decoder_attention_mask_input[:,2:]


        loss_fct = nn.CrossEntropyLoss(reduce=False)
        if self.type == "test":
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size), decoder_input_ids_gt.view(-1))
            losses_resize = losses.view(decoder_attention_mask_input.size())
            return losses_resize * decoder_attention_mask_input.float(), lm_logits
        
        losses_lm = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids_gt.reshape(-1))

        loss_lm = torch.sum(losses_lm * decoder_attention_mask_gt.float().view(-1))
        
        
        clf_logits = self.classification_head(outputs[0][:,0])
        losses_clf = F.cross_entropy(clf_logits.view(-1, self.model_cfg.video_type_num), video_types.view(-1), weight = self.clf_loss_weight)

        loss_clf = torch.sum(losses_clf)
        

        loss = loss_lm + self.model_cfg.clf_weight * loss_clf

        return loss, (lm_logits,clf_logits)
    
    def attach_visual_for_mask(self,attention_mask):
        
        visual = self.encoder.visual
        visual_ones = torch.ones([visual.size()[0],visual.size()[1]], dtype=attention_mask.dtype).to(torch.device("cuda"))
        attention_mask_cat = torch.cat([visual_ones,attention_mask], dim=1) 
        return(attention_mask_cat)
    
 
class MyClassificationPLVCG(MyPLVCG):
    def __init__(self, model_cfg, negative_num=1, type="classification"):
        super().__init__(model_cfg, None, type)
        self.negative_num = negative_num-1
        self.loss_weight = torch.from_numpy(np.array([1.0,self.negative_num*1.0])).float().to(torch.device("cuda"))
        self.rank_classification_head = BartClassificationHead(
            self.config.d_model, self.config.d_model, self.config.num_labels, 0.0,
        )
        self.model._init_weights(self.rank_classification_head.dense)
        self.model._init_weights(self.rank_classification_head.out_proj)
        
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, labels=None, decoder_cached_states=None,
            use_cache=False, is_training=False):


        decoder_attention_mask_input = decoder_attention_mask
        
        outputs = self.model(
            input_ids,
            attention_mask=self.attach_visual_for_mask(attention_mask),
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask_input,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        
        x = outputs[0]  # last hidden state
        eos_mask = decoder_input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.rank_classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1), weight = self.loss_weight)

        if self.type == "test":
            loss = None

        return loss, logits