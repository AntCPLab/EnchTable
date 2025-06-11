import os
import time
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import sys
import logging
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"

class SafeDecoding():
    pass
    def __init__(self, model_path, expert_model_path, max_new_tokens=256, first_m = 2, top_k = 10, num_common_tokens = 5, alpha = 4, do_sample = False, temperature = 0.00, top_p = 1.0, verbose = True, batch_size = 8, system_prompt = None):
        self.model_name = model_path
        self.expert_model_name = expert_model_path
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.first_m = first_m
        self.top_k   = top_k
        self.num_common_tokens = num_common_tokens
        self.alpha = alpha
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def generate(self, prompt):

        class SafeDecodingDataset(torch.utils.data.Dataset):

            def __init__(self, data_list, tokenizer, system_prompt=None):
                ## data_list [{"question": ..., "input":, ...,}]
                self.data_list = data_list
                self.tokenizer = tokenizer
                self.features = []
                truncation_count = 0
                for idx, data in enumerate(self.data_list):
                    
                    data_item = tokenizer.apply_chat_template(
                                data,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            ).to(device)
                    attention_mask = torch.ones_like(data_item)
                    feature = {
                        "input_ids": data_item.squeeze(0),
                        "attention_mask": attention_mask.squeeze(0),
                    }
                    # import pdb; pdb.set_trace()
                    # data_item = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False, truncation=True, max_length = 1500)
                    # data_b = self.tokenizer(chat_template, padding=True, return_tensors="pt", add_special_tokens = False, truncation=True)
                    # if data_item['input_ids'].shape[-1] < data_item["input_ids"].shape[-1]:
                    #     truncation_count += 1
                    # features
                    # data_item = {k: v[0] for k, v in data_item.items()}
                    if idx == 0:
                        pass
                        decode_text = self.tokenizer.decode(feature["input_ids"].tolist(), skip_special_tokens = False)
                        print(f"chat_template:{[decode_text]}")
                    
                    self.features.append(feature)
                print(f"truncation_count: {truncation_count}")

            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, index):
                return self.features[index]

        dataset = SafeDecodingDataset(data_list=prompt, tokenizer= self.tokenizer, system_prompt = self.system_prompt)
        data_collator = DataCollatorWithPadding(self.tokenizer, padding = "longest")
        tmp_dataloader = DataLoader(dataset, batch_size = self.batch_size, collate_fn=data_collator)

        class StopAtSpecificTokenCriteria(StoppingCriteria):
            def __init__(self, token_id_list, batch_size):
                # import pdb; pdb.set_trace()
                self.token_id_list = token_id_list
                self.count_eos_batch = [False for _ in range(batch_size)]
            
            def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
                # import pdb; pdb.set_trace()
                result = True
                for i, tmp_idx in enumerate(input_ids):
                    if tmp_idx[-1].detach().cpu().numpy() in self.token_id_list:
                        self.count_eos_batch[i] = True
                for eos in self.count_eos_batch:
                    result = result and eos
                return result



        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype="auto",)
        expert_model = AutoModelForCausalLM.from_pretrained(self.expert_model_name, device_map="auto", torch_dtype="auto",)

        model.eval()
        expert_model.eval()

        gen_config = model.generation_config
        gen_config.max_new_tokens = 1
        gen_config.do_sample = False
        self.verbose = True
        time_start = time.time()
        pass
        safe_decoding_prompts_idx = []

        # orig_input_lens = []
        for batch in tqdm(tmp_dataloader, total=len(tmp_dataloader)):
            stopping_criteria = []
            tmp_batch_size = batch['input_ids'].shape[0]
            seq_len = batch['input_ids'].shape[1]
            # orig_input_lens.extend([seq_len] * tmp_batch_size)
            if "llama-3" in self.model_name.lower() or "llama3" in self.model_name.lower():
                # stop_ids.append(128009)
                # stop_criteria = StoppingCriteria(stop_ids)
                # stopping_criteria.append(StoppingCriteria([tokenizer.convert_tokens_to_ids("<|eot_id|>")], batch_size))
                stopping_criteria = StoppingCriteriaList()
                # stopping_criteria.append(StopAtSpecificTokenCriteria(tokenizer.eos_token_id], tmp_batch_size))
                stopping_criteria.append(StopAtSpecificTokenCriteria([self.tokenizer.convert_tokens_to_ids("<|eot_id|>")], tmp_batch_size))
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to("cuda")
                step = 1
                while step <= min(self.max_new_tokens, self.first_m):
                    # import pdb; pdb.set_trace()
                    # data_batch_size = batch_size['input_ids'].shape[0]
                    output_base = model.generate(**batch,
                        # adapter_names=self.adapter_names,
                        generation_config=gen_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,)
                    # import pdb; pdb.set_trace()
                    output_expert = expert_model.generate(**batch,
                        # adapter_names=self.adapter_names,
                        generation_config=gen_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,)
                    ##
                    # output_base 
                    #   sequences: ( batch_size, input_seq + 1 ) 
                    #   scores: ( tensor( batch_size, vocab_size ), ... ( * max_new_tokens), )
                    ##
                    k = self.top_k
                    scores_base = output_base.scores[-1]
                    scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
                    topk_scores_base, topk_indices_base = scores_base.topk(k, dim=-1) 
                    # import pdb; pdb.set_trace()

                    scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
                    scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
                    topk_scores_expert, topk_indices_expert = scores_expert.topk(k, dim=-1) 

                    sorted_indices_base = torch.argsort(scores_base, descending=True, dim=-1)
                    sorted_indices_expert = torch.argsort(scores_expert, descending=True, dim=-1)

                    # Step 1: Define Sample Space
                    
                    num_common_tokens = self.num_common_tokens
                    batch_select_token_id = []
                    for bsz_idx in range(tmp_batch_size):
                        
                        # import pdb; pdb.set_trace()
                        common_tokens = set()
                        iter_range = num_common_tokens
                        while len(common_tokens) < num_common_tokens:
                            current_indices_base = sorted_indices_base[bsz_idx][:iter_range]
                            current_indices_expert = sorted_indices_expert[bsz_idx][:iter_range]

                            common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                            common_tokens.update(common_in_iteration)
                            # print("common_tokens: {}".format(common_tokens))
                            # print("update: {}".format(common_in_iteration))

                            iter_range += 1

                            if iter_range > min(len(sorted_indices_base[bsz_idx]), len(sorted_indices_expert[bsz_idx])):
                                break
                        
                        # import pdb; pdb.set_trace()
                        # Display the top tokens
                        if self.verbose and step == 1:
                            logging.info("\n-----------------------------------------------")
                            logging.info(f"Generation Step {step}")
                            logging.info("Original Model")
                            logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                            logging.info("|----|----------|---------|----------|---------|")
                            for idx, (score, token_id) in enumerate(zip(topk_scores_base[bsz_idx], topk_indices_base[bsz_idx])):
                                token = self.tokenizer.decode(token_id.item())
                                prob = torch.exp(score)
                                logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                            logging.info("Expert Model")
                            logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                            logging.info("|----|----------|---------|----------|---------|")
                            for idx, (score, token_id) in enumerate(zip(topk_scores_expert[bsz_idx], topk_indices_expert[bsz_idx])):
                                token = self.tokenizer.decode(token_id.item())
                                prob = torch.exp(score)
                                logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                        # import pdb; pdb.set_trace()
                        intersection_indices = torch.tensor(list(common_tokens), device=model.device)
                        # import pdb; pdb.set_trace()
                        # Step 2: New Probability Calculation
                        updated_scores = []
                        for token_id in intersection_indices:
                            # Steer scores
                            # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                            # updated_scores.append(new_score)

                            # Steer probabilities
                            prob_diff = torch.exp(scores_expert[bsz_idx][token_id]) - torch.exp(scores_base[bsz_idx][token_id])
                            updated_prob = torch.exp(scores_base[bsz_idx][token_id]) + self.alpha * prob_diff
                            # Floor the probability to 1e-8 to avoid log(0)
                            updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=model.device)
                            updated_score = torch.log(updated_prob)
                            updated_scores.append(updated_score)

                            if self.verbose:
                                logging.info(f"----------------token id: {token_id}-----------------")
                                logging.info(f"Prob Base: {torch.exp(scores_base[bsz_idx][token_id])}")
                                logging.info(f"Prob Expert: {torch.exp(scores_expert[bsz_idx][token_id])}")
                                logging.info(f"Base score: {scores_base[bsz_idx][token_id]}")
                                logging.info(f"Expert score: {scores_expert[bsz_idx][token_id]}")
                                logging.info(f"Updated Probability: {updated_prob}")
                                logging.info(f"Updated Score: {updated_score}")

                        # Use softmax to normalize the scores
                        # This is to ensure that the probability sum to 1
                        # import pdb; pdb.set_trace()
                        normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

                        sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
                        sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
                        sorted_token_ids = [intersection_indices[i] for i in sorted_indices]
                        # import pdb; pdb.set_trace()
                        if self.verbose:
                            logging.info("\n-----------------------------------------------")
                            logging.info(f"Generation Step {step}")
                            logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                            logging.info("|----|----------|---------|----------|---------|")
                            for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                                token = self.tokenizer.decode(token_id.item())
                                score = torch.log(prob)
                                logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                        ### Sample the next token
                        if self.do_sample == False:
                            # Greedy decoding
                            # Append the selected token to the sequence
                            selected_token_id = sorted_token_ids[0].unsqueeze(0)
                        elif gen_config.top_p != None and self.do_sample == True:
                            # Top-p sampling, sample from the top-p tokens
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            p_index = torch.where(cumulative_probs >= gen_config.top_p)[0][0]
                            sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                            sorted_top_p_probs = sorted_probs[:p_index + 1]
                            sorted_top_p_scores = torch.log(sorted_top_p_probs)
                            if self.verbose:
                                logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                                logging.info(f"Top-p scores: {sorted_top_p_scores}")
                                logging.info(f"Top-p probabilities: {sorted_top_p_probs}")
                            
                            # Sample from the top-p tokens
                            selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
                        else:
                            raise ValueError("Please set do_sample to False or top_p to a value.")

                        if self.verbose:
                            logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
                        # generated_sequence.append(selected_token_id.item())

                        # if the chosen token id is eos, then stop
                        if selected_token_id.item() == self.tokenizer.eos_token_id:
                            # selected_token_id = p
                            pass
                            # break
                        batch_select_token_id.append(selected_token_id)

                    # import pdb; pdb.set_trace()
                    batch_select_token_id = torch.tensor(batch_select_token_id).to(model.device).unsqueeze(1)
                    batch['input_ids'] = torch.cat([batch['input_ids'], batch_select_token_id], dim=1)
                    batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(batch_select_token_id, device=model.device)], dim=1)
                    step += 1
                    del output_base, output_expert

            safe_decoding_prompts_idx.extend(batch['input_ids'].tolist())

        # import pdb; pdb.set_trace()

        safe_decoding_prompts = []
        for prompt_tokens in safe_decoding_prompts_idx:
            decode_prompts = self.tokenizer.decode(prompt_tokens, skip_special_token = False).lstrip(self.tokenizer.pad_token)
            safe_decoding_prompts.append(decode_prompts)

        del expert_model, model
        ###setting up model###
        llm = LLM(self.model_name)

        temperature = self.temperature
        top_p = self.top_p
        max_tokens = self.max_new_tokens
        prompts = safe_decoding_prompts

        print("----------------------------------------------------")
        print("input:\n{}".format(prompts[0]))
        print("----------------------------------------------------")

        stop_words = ["</s>"]
        # if "llama3" in self.model_name.lower() or "llama-3" in self.model_name.lower():
        #     stop_words.append("<|eot_id|>")
        outputs = llm.generate(prompts, SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=1,
                        stop=stop_words,
        ))
        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        outputs_list = [output.outputs[0].text for output in outputs]
        # import pdb;pdb.set_trace()
        new_output_list = []
        print("----------------------------------------------------")
        print("output:\n{}".format(outputs_list))
        for prompt, response in zip(safe_decoding_prompts, outputs_list):
            prompt_response_pair = prompt + response
            if "llama3" in self.model_name.lower() or "llama-3" in self.model_name.lower():
                pass
                response = prompt_response_pair.split("assistant<|end_header_id|>")[-1]
                # import pdb;pdb.set_trace()
            else:
                response = prompt_response_pair.split("[/INST]")[-1]
            pass
            # unwanted_tokens = ["<|eot_id|>", "<|start_header_id|>"]
            # for token in unwanted_tokens:
            #     response = response.replace(token, "")
            new_output_list.append(response)
        outputs_list = new_output_list
        # import pdb;pdb.set_trace()
        return outputs_list

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Initialize SafeDecoding
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Example model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
    
    tokenizer.pad_token = tokenizer.eos_token
    safe_decoder = SafeDecoding(
            model_path="./llama3-8b-beavertail_harmful",
            expert_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
            verbose=True,
            batch_size=4,
        )


    # Input batch of messages
    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How does machine learning work?"},
        ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "who are you?"},
            ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather like today?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the best programming language?"},
        ],
    ]
    # Generate responses using SafeDecoding
    responses = safe_decoder.generate(batch_messages)

    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")