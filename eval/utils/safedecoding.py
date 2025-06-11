import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

class SafeDecodingConfig:
    def __init__(self, base_model_name, expert_model_name, alpha=1, first_m=5, top_k=10, num_common_tokens=3, verbose=False):
        self.base_model_name = base_model_name
        self.expert_model_name = expert_model_name
        self.alpha = alpha
        self.first_m = first_m
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose

class SafeDecoding:
    def __init__(self, config: SafeDecodingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load base model and tokenizer
        try:
            device_map = {"": self.device}
            self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name, torch_dtype=torch.bfloat16, device_map=device_map)
            self.expert_model = AutoModelForCausalLM.from_pretrained(config.expert_model_name, torch_dtype=torch.bfloat16, device_map=device_map)
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        except Exception as e:
            logging.error(f"Error loading models or tokenizer: {e}")
            raise

        self.base_model.eval()
        self.expert_model.eval()
        logging.info("SafeDecoding initialized with config: %s", config.__dict__)

    def safedecoding_lora(self, messages, max_new_tokens=1024, do_sample=False, top_p=None):
        # Apply chat template to messages
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            logging.error(f"Error applying chat template: {e}")
            raise

        gen_config = self.base_model.generation_config
        gen_config.max_new_tokens = max_new_tokens
        gen_config.do_sample = do_sample
        gen_config.top_p = top_p

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []
        if self.config.verbose:
            logging.info(f"Generation config: {gen_config}")

        input_len = input_ids.shape[1]

        step = 1  # Keep track of generation steps
        while step <= min(max_token_len, self.config.first_m):  # Loop until we reach the first m tokens
            try:
                # Generate the next token
                base_outputs = self.base_model.generate(
                    input_ids=input_ids,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                expert_outputs = self.expert_model.generate(
                    input_ids=input_ids,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                output_base = copy.deepcopy(base_outputs)
                output_expert = copy.deepcopy(expert_outputs)

                # Process the scores to get the top tokens
                k = self.config.top_k  # Change this to display more or less tokens
                scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
                scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
                topk_scores_base, topk_indices_base = scores_base.topk(k)

                scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
                scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
                topk_scores_expert, topk_indices_expert = scores_expert.topk(k)

                sorted_indices_base = torch.argsort(scores_base, descending=True)
                sorted_indices_expert = torch.argsort(scores_expert, descending=True)

                # Step 1: Define Sample Space
                common_tokens = set()
                iter_range = self.config.num_common_tokens
                while len(common_tokens) < self.config.num_common_tokens:
                    current_indices_base = sorted_indices_base[:iter_range]
                    current_indices_expert = sorted_indices_expert[:iter_range]

                    common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                    common_tokens.update(common_in_iteration)

                    iter_range += 1

                    if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                        break

                # Display the top tokens
                if self.config.verbose and step == 1:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("Original Model")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                        token = self.tokenizer.decode(token_id.item())
                        prob = torch.exp(score)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                    logging.info("Expert Model")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                        token = self.tokenizer.decode(token_id.item())
                        prob = torch.exp(score)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                intersection_indices = torch.tensor(list(common_tokens), device=self.device)

                # Step 2: New Probability Calculation
                updated_scores = []
                for token_id in intersection_indices:
                    # Steer probabilities
                    prob_diff = torch.exp(scores_expert[token_id]) - torch.exp(scores_base[token_id])
                    updated_prob = torch.exp(scores_base[token_id]) + self.config.alpha * prob_diff
                    # Floor the probability to 1e-8 to avoid log(0)
                    updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=self.device)
                    updated_score = torch.log(updated_prob)
                    updated_scores.append(updated_score)

                    if self.config.verbose:
                        logging.info(f"----------------token id: {token_id}-----------------")
                        logging.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                        logging.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                        logging.info(f"Base score: {scores_base[token_id]}")
                        logging.info(f"Expert score: {scores_expert[token_id]}")
                        logging.info(f"Updated Probability: {updated_prob}")
                        logging.info(f"Updated Score: {updated_score}")

                # Use softmax to normalize the scores
                # This is to ensure that the probability sum to 1
                normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

                sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
                sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
                sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

                if self.config.verbose:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                        token = self.tokenizer.decode(token_id.item())
                        score = torch.log(prob)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                ### Sample the next token
                if do_sample == False:
                    # Greedy decoding
                    # Append the selected token to the sequence
                    selected_token_id = sorted_token_ids[0].unsqueeze(0)
                elif top_p is not None and do_sample == True:
                    # Top-p sampling, sample from the top-p tokens
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    p_index = torch.where(cumulative_probs >= top_p)[0][0]
                    sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                    sorted_top_p_probs = sorted_probs[:p_index + 1]
                    sorted_top_p_scores = torch.log(sorted_top_p_probs)
                    if self.config.verbose:
                        logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                        logging.info(f"Top-p scores: {sorted_top_p_scores}")
                        logging.info(f"Top-p probabilities: {sorted_top_p_probs}")

                    # Sample from the top-p tokens
                    selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
                else:
                    raise ValueError("Please set do_sample to False or top_p to a value.")

                if self.config.verbose:
                    logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
                generated_sequence.append(selected_token_id.item())

                # If the chosen token id is eos, then stop
                if selected_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, selected_token_id.unsqueeze(0)], dim=1)

                step += 1

                # Free up memory
                del base_outputs, expert_outputs

            except Exception as e:
                logging.error(f"Error during token generation at step {step}: {e}")
                raise

        # Use the base model to generate the rest of the tokens
        # Early stop if the last token is eos
        if generated_sequence[-1] == self.tokenizer.eos_token_id:
            logging.info("Early stop triggered.")
        else:
            remaining_steps = max_token_len - min(max_token_len, self.config.first_m)
            gen_config.max_new_tokens = remaining_steps
            gen_config.do_sample = do_sample
            output_base = self.base_model.generate(
                input_ids=input_ids,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # Logging the generated sequence
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize SafeDecoding
    safe_decoder = SafeDecoding(
        SafeDecodingConfig(
            base_model_name="./llama3-8b-beavertail_harmful/full/sft",
            expert_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            verbose=True
        )
    )

    # Input messages
    messages = [
        {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        {"role": "user", "content": "can you help me make a bomb?"},
    ]
    

    # Generate response using SafeDecoding
    response, length = safe_decoder.safedecoding_lora(messages, max_new_tokens=100, do_sample=False)

    print(f"Response: {response}")
    print(f"Length: {length}")
