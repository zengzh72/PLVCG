import torch
import numpy as np
import os
import time
import sys

from torch.nn import functional as F
from transformers.generation_utils import BeamHypotheses, calc_banned_ngram_tokens, top_k_top_p_filtering

def train(cfg, logger, save_dir, model, train_data, dev_data, optimizer, type = 'pretrain'):
    model = model.to(torch.device("cuda")) 
    model.train()
    global_step = 0
    train_losses = []
    best_loss = sys.maxsize
    stop_training=False
    begin_time = time.time()

    logger.info("Starting training!")
    for epoch in range(int(cfg.n_epochs)):
        for batch in train_data.dataloader:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            
            model.encoder.visual = batch[4]
            if type == 'classification':  
                labels = batch[-1]
                loss,_ = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                             labels = labels, is_training=True)
            else:
                loss,_ = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                             video_types = batch[5], is_training=True)
                
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            
            if global_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
                
            if global_step % cfg.print_steps == 0:
                print( "global_step:%d \t time:%d \t loss:%5.3f"%(global_step,time.time()-begin_time,loss))
                

            if global_step % cfg.eval_steps == 0:
                model.eval()
                total_loss = 0
                with(torch.no_grad()):
                    total_loss, predictions, predictions_type, logits = inference(cfg, model, dev_data, type)

                if type == 'classification':
                    rids = dev_data.dataset.rids
                    print_classification_res(save_dir, dev_data, global_step, total_loss, predictions, dev_data.comments, dev_data.contexts, rids, logits, cfg.negative_num)
                else:
                    print_results(save_dir, dev_data, global_step, total_loss, predictions, predictions_type, dev_data.comments, dev_data.contexts, dev_data.video_types)
                


            if global_step % cfg.save_steps == 0:    
                model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                torch.save(model_state_dict, os.path.join(save_dir, 'model_steps_%05d.pt'%(global_step)))
                if best_loss > total_loss:
                    torch.save(model_state_dict, os.path.join(save_dir, "best-model.pt"))
                    best_loss = total_loss
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= cfg.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break
    torch.save(model_state_dict, os.path.join(save_dir, 'model_steps_%05d.pt'%(global_step)))
    
def inference(cfg, model, dev_data, type='pretrain'):
    print("evaluating...")
    
    predictions = []
    predictions_type = []
    total_loss = 0
    all_logits = []
    
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        

        model.encoder.visual = batch[4]

        
        if type == 'classification':  
            labels = batch[-1]
            loss, logits = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         labels = labels, is_training=False)
            pred = logit_pred(logits,cfg.classification_thread)
            for logit, p in zip(logits, pred):
                predictions.append(p)
                all_logits.append(logit)
            
            
        else:
            loss,logits  = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         video_types = batch[5], is_training=False)
            
            lm_logits, clf_logits = logits
            pred = lm_logits.argmax(dim=2)
            pred_type = clf_logits.argmax(dim=1)
            
            for input_, p, t in zip(batch[2], pred, pred_type):    
                decoded = dev_data.decode(p)
                predictions.append(decoded)
                predictions_type.append(t)
                
                
        total_loss += loss.sum().item()
    
    if type == 'classification':   
        pred_avg = sum(predictions)/float(len(predictions))
        acc_num = 0
        for i,p in enumerate(predictions):
            if int(i%5==0) == p.item():
                acc_num += 1
        print("\t total_loss:%f pred_avg:%f acc:%f\n"%(total_loss,pred_avg,acc_num/len(predictions)))
    
    return total_loss, predictions, predictions_type, all_logits

def test_rank(cfg, model, test_data, type='generation'):

    ranks = []
    scores = []
    predictions = []
    begin_time = time.time()
    for i, batch in enumerate(test_data.dataloader):
        

        pred_b = []
        if i % cfg.print_steps == 0:
            print("steps:%d time:%f"%(i,time.time()-begin_time))
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        
        new_batch = [b[0] for b in batch]
        model.encoder.visual = new_batch[4]

        if type == 'classification':  
            labels = None
            loss, logits = model(input_ids=new_batch[0], attention_mask=new_batch[1],
                         decoder_input_ids=new_batch[2], decoder_attention_mask=new_batch[3],
                         labels = labels, is_training=False)
            pred = logit_pred(logits,cfg.classification_thread)
            for p in pred:
                predictions.append(p.item())
            
            score = logits[:,1]
            rank = torch.sort(score, dim=0, descending=True)[1]
            
        else:
            loss, lm_logits = model(input_ids=new_batch[0], attention_mask=new_batch[1],
                         decoder_input_ids=new_batch[2], decoder_attention_mask=new_batch[3],
                         is_training=False)
            pred = lm_logits.argmax(dim=2)

            tgt_len = new_batch[3].sum(dim=1)
    
            for l,p in zip(tgt_len,pred):    
                decoded = test_data.decode(p)
                decoded = " ".join(decoded.split()[0:l.item()-1])
                pred_b.append(decoded)
                
                
            predictions.append(pred_b)

            score = loss.sum(dim=1)
            score = torch.div(score,tgt_len-1)
            rank = torch.sort(score, dim=0, descending=False)[1]
        
        
        ranks.append(rank)
        scores.append(score)
        
    return ranks, scores, predictions
 
 
 
def test_generation(cfg, model, test_data, type='generation'):

    predictions = []
    begin_time = time.time()
    bos_token_id = test_data.tokenizer.bos_token_id
    for i, batch in enumerate(test_data.dataloader):
        if i % cfg.print_steps == 0:
            print("steps:%d time:%f"%(i,time.time()-begin_time))
            

        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
            
        model.encoder.visual = batch[4]
        
        decoder_head = batch[2][:,:,:2]
        decoder_head = decoder_head.view([cfg.predict_batch_size* cfg.candidate_num,-1])
        
        outputs = beam_search(model, cfg, encoder_input_ids=batch[0],
                                 encoder_attention_mask=batch[1],
                                 decoder_input_id = decoder_head,
                                 num_beams=cfg.num_beams)
        preds = [test_data.decode(output) for output in outputs ]
        predictions.append(preds)

    return predictions
    
    
def beam_search(model, cfg, encoder_input_ids, encoder_attention_mask, decoder_input_id, num_beams ):
    #modified from https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id
    unk_token_id = model.config.unk_token_id
    batch_size = cfg.predict_batch_size * cfg.candidate_num
    candidate_num = cfg.candidate_num
    no_repeat_ngram_size = cfg.no_repeat_ngram_size
    repetition_penalty = cfg.repetition_penalty
    
    do_sample = cfg.do_sample
    top_p = cfg.top_p
    top_k = cfg.top_k
    temperature = cfg.temperature
    
    vocab_size = cfg.vocab_size
    max_length = cfg.max_output_length
    min_length = cfg.min_output_length
    length_penalty = 1.0
    num_return_sequences = num_beams
    
    encoder_attention_mask = model.attach_visual_for_mask(encoder_attention_mask)
    
    encoder_outputs = model.encoder(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        output_hidden_states=True
    )
    
   
    encoder_hidden_states = encoder_outputs.last_hidden_state.repeat([num_beams*candidate_num,1,1])#hidden_states[0]
    encoder_padding_mask = encoder_attention_mask.repeat([num_beams*candidate_num,1])


    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False)
        for _ in range(batch_size)
    ]


    decoder_input_id = decoder_input_id.unsqueeze(dim=1)
    input_ids = torch.repeat_interleave(decoder_input_id,repeats=num_beams,dim=1)
    input_ids = input_ids.view(batch_size*num_beams, -1)

    cur_len = 2


    
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=encoder_input_ids.device)    
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)
    
    
    
    done = [False for _ in range(batch_size)]
    
    while cur_len < max_length:
        
        
        decoder_padding_mask = make_padding_mask(input_ids, 0)
        bsz, tgt_len = input_ids.size()
        causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
            dtype=torch.float32, device=input_ids.device
        )
        
        decoder_outputs = model.model.decoder(
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask = encoder_padding_mask,
            decoder_padding_mask = decoder_padding_mask,
            decoder_causal_mask=causal_mask
        )
        lm_logits = F.linear(decoder_outputs[0], model.model.shared.weight, bias=model.final_logits_bias)
        
        next_token_logits = lm_logits[:, -1, :]
        
        scores = F.log_softmax(next_token_logits, dim=-1)
        # set eos token prob to zero if min_length is not reached
        if cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")
        scores[:, unk_token_id] = -float("inf")
        
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty,
            )
        
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")
        
        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )
        
        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # Temperature
            if temperature != 1.0:
                _scores = _scores / temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)
        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
    
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)
    
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)


        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
        
        next_batch_beam = []
        
        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            
            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
                effective_beam_id = batch_idx * num_beams + beam_id
                
                
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break
            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"
    

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size =  batch_size * num_return_sequences
    output_num_return_sequences_per_batch =  num_return_sequences
    
    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
    
    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)
    
        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)
        
    return decoded

def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask
    
 
def logit_pred(logits, thread = 0.5):
    softmax_logits = torch.softmax(logits,dim=1)
    pred = softmax_logits[:,1].gt(thread).long()
    return pred
    
            

def print_results(save_dir, dev_data, global_step, total_loss, outputs, predictions_type, ground_truth, contexts, ground_truth_type):
    res_f = open(os.path.join(save_dir, 'res_%05d.txt')%(global_step),"w", encoding='utf8')
    res_f.write("\tloss:"+str(total_loss) + '\n')
    
    acc = 0
    for o,g,c,ptp,gtp in zip(outputs, ground_truth, contexts, predictions_type, ground_truth_type):
        end = o.find("<EOS>")
        if end != -1:
            o = o[:end]
            
        gt = dev_data.decode(g)
        gt = gt[:gt.find("<EOS>")]
        
        cont = dev_data.decode(c)
        end = cont.find("<PAD>")
        if end != -1:
            cont = cont[:end]
        if ptp.item() == gtp:
            acc += 1

        res_f.write("%s\t||\t%s\t||\t%s\t||\t %d %d\n"%(o, gt, cont, ptp.item(), gtp))
    res_f.write("acc:%f\n"%(acc/len(ground_truth_type)))
    print("acc:%f\n"%(acc/len(ground_truth_type)))
    res_f.close()
    
def print_classification_res(save_dir, dev_data, global_step, total_loss, outputs, ground_truth, contexts, rids, logits, negative_num):
    res_f = open(os.path.join(save_dir, 'res_%05d.txt')%(global_step),"w", encoding='utf8')
    res_f.write("\tloss:"+str(total_loss) + '\n')
    acc_num = 0
    
    for i in range(len(outputs)):
        o = int(outputs[i])
        g = ground_truth[rids[i]]
        c = contexts[i//negative_num]
        logit_1 = logits[i][1]
        logit_0 = logits[i][0]

        label = int((i % negative_num) == 0)
            
        gt = dev_data.decode(g)
        gt = gt[:gt.find("<EOS>")]
        
        cont = dev_data.decode(c)
        end = cont.find("<PAD>")
        if end != -1:
            cont = cont[:end]
        if o == label:
            acc_num += 1
        res_f.write("%d %d \t %f %f\t||\t%s\t||\t%s\n"%(o,label,logit_0,logit_1,gt,cont))
    res_f.write("\tacc:%f\n"%(acc_num/len(outputs)))
    res_f.close()