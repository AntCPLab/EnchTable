import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using LLaMA3 model.")
    parser.add_argument(
        "--input_file", type=str, default="./harmful_questions/sampled_salad_dataset.json", help="Path to the local JSON file containing the dataset"
    )
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Path to the LLaMA3 model"
    )
    parser.add_argument(
        "--generation_method", type=str, default="sample", choices=["greedy", "sample"], help="Generation method"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p value for sampling"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generating responses"
    )
    parser.add_argument('--save_path', help='Path where the model results will be saved', type=str, default='./test_salad')
    parser.add_argument('--llama2', action='store_true', help='Use Llama2 model')

    return parser.parse_args()

def simplify_model_id(model_id):
    if '/' in model_id:
        # Remote model, take the last part
        simplified_id = model_id.split('/')[-1]
    else:
        # Local model, take the last directory or filename
        simplified_id = os.path.basename(model_id)
    return simplified_id

def generate_batch_responses(questions, tokenizer, model, generation_method='sample', temperature=0.6, top_p=0.9):
    messages = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
        for question in questions
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device) # if batch_size > 1
    # input_ids = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
    ]
    
    kwargs = {
        'max_new_tokens': 256,
        'eos_token_id': terminators,
        'temperature': temperature,
        'top_p': top_p
    }
    
    if generation_method == 'greedy':
        kwargs['do_sample'] = False
    elif generation_method == 'sample':
        kwargs['do_sample'] = True

    # if batch_size == 1
    kwargs['attention_mask'] = torch.ones_like(input_ids)
    
    outputs = model.generate(**kwargs, input_ids=input_ids)
    responses = [tokenizer.decode(output[input_ids[i].shape[-1]:], skip_special_tokens=True) for i, output in enumerate(outputs)]
    return responses

def generate_responses_llama2(prompt_que, tokenizer, model, batch_size):
    outputs_list = []
    for i in range(0, len(prompt_que), batch_size):
        batch_prompts = prompt_que[i:i + batch_size]
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # input_ids = tokenizer.apply_chat_template(
        # messages,
        # add_generation_prompt=True,
        # return_tensors="pt").to(model.device)
        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
    ]
        kwargs = {
            'eos_token_id': terminators
        }
        outputs_ge = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding
                **kwargs
            ) # batch_size == 1
        
        responses = tokenizer.batch_decode(outputs_ge, skip_special_tokens=True)
        for j, response in enumerate(responses):
            response = response.replace(batch_prompts[j], '').strip()
            outputs_list.append(response)
    return outputs_list

def process_dataset(dataset, tokenizer, model, generation_method='sample', temperature=0.6, top_p=0.9, batch_size=4, llama2=False):
    results = []
    total_questions = len(dataset)
    exception_count = 0
    
    for i in tqdm(range(0, total_questions, batch_size), desc="Generating responses"):
        batch = dataset[i:i + batch_size]
        questions = [entry['question'] for entry in batch]
        
        try:
            if llama2:
                # print(f"User Llama2 model to generate responses")
                answers = generate_responses_llama2(questions, tokenizer, model, batch_size)
            else:
                # Generate responses for the batch
                answers = generate_batch_responses(questions, tokenizer, model, generation_method, temperature, top_p)
            
            for j, entry in enumerate(batch):
                qid = entry.get('qid', None)
                source = entry.get('source', None)
                question = entry['question']
                label = entry['1-category']
                answer = answers[j]
                
                result_entry = {
                    'qid': qid,
                    'source': source,
                    'question': question,
                    'answer': answer,
                    '1-category': label
                }
                results.append(result_entry)
        except Exception as e:
            logging.error(f"Exception occurred while processing batch starting at question: {batch[0]['question']}. Error: {e}")
            for entry in batch:
                qid = entry.get('qid', None)
                source = entry.get('source', None)
                question = entry['question']
                label = entry['1-category']
                
                result_entry = {
                    'qid': qid,
                    'source': source,
                    'question': question,
                    'answer': str(e),
                    '1-category': label
                }
                results.append(result_entry)
                exception_count += 1
    
    overall_stats = {
        'total_questions': total_questions,
        'exception_count': exception_count
    }
    
    return results, overall_stats

def save_results(results, stats, output_path):
    with open(output_path, 'w') as f:
        json.dump({'results': results, 'stats': stats}, f, indent=4)

def main():
    args = parse_args()
    
    # Suppress specific warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Set up logging to file
    logging.basicConfig(filename='generate_responses.log', level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load dataset from local JSON file
    with open(args.input_file, 'r') as f:
        dataset = json.load(f)
    
    # Initialize the tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    results, stats = process_dataset(dataset, tokenizer, model, args.generation_method, args.temperature, args.top_p, args.batch_size, args.llama2)
    
    simplified_model_id = simplify_model_id(args.model_id)
    output_path = os.path.join(args.save_path, f"{simplified_model_id}_response.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(results, stats, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
