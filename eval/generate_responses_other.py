import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_dataset(file_path, num_samples, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'harmfulq' in file_path or 'cat' in file_path:
        topics = []
        subtopics = []
        prompt_que = []
        orig_que = []
        for topic in data.keys():
            for subtopic in data[topic].keys():
                for q in data[topic][subtopic]:
                    orig_que.append(q)
                    prompt_que.append(q)
                    topics.append(topic)
                    subtopics.append(subtopic)
    else:
        prompt_que = [q for q in data]
        orig_que = data
        topics, subtopics = [], []

    if num_samples == -1:
        num_samples = len(prompt_que)

    return prompt_que[:num_samples], orig_que[:num_samples], topics[:num_samples], subtopics[:num_samples]

def generate_responses(prompt_que, tokenizer, model, batch_size):
    outputs = []
    for i in tqdm(range(0, len(prompt_que), batch_size)):
        batch_prompts = prompt_que[i:i + batch_size]

        messages = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
        for question in batch_prompts
    ]
        # input_ids = tokenizer.apply_chat_template(
        # messages,
        # add_generation_prompt=True,
        # return_tensors="pt").to(model.device)
        input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt").to(model.device) # batch_size == 1
        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
    ]
        kwargs = {
            'max_new_tokens': 256,
            'eos_token_id': terminators,
            'temperature': 0.6,
            'top_p': 0.9
        }
        kwargs['do_sample'] = False
        # bathch_size = 1
        kwargs['attention_mask'] = torch.ones_like(input_ids)
        outputs_ge = model.generate(**kwargs, input_ids=input_ids)
        responses = [tokenizer.decode(output[input_ids[i].shape[-1]:], skip_special_tokens=True) for i, output in enumerate(outputs_ge)]
        for j, response in enumerate(responses):
            outputs.append((batch_prompts[j], response))

    return outputs

def generate_responses_llama2(prompt_que, tokenizer, model, batch_size):
    outputs_list = []
    for i in tqdm(range(0, len(prompt_que), batch_size)):
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
            outputs_list.append((batch_prompts[j], response))
            


    return outputs_list

def clean_thoughts(response):
    if "(Internal thought:" in response:
        start = response.find("(Internal thought:") + len("(Internal thought:")
        end = response.find(")", start)
        return response[end + 1:].strip() if end != -1 else response[start:].strip()
    return response

def process_responses(outputs, orig_que, topics, subtopics, dataset):
    processed_outputs = []
    # import pdb;pdb.set_trace()
    for i, (prompt, response) in enumerate(outputs):
        response = clean_thoughts(response.replace(prompt.replace('<s>', ''), '').strip())
        if 'harmfulq' in dataset or 'cat' in dataset:
            processed_outputs.append({
                'prompt': orig_que[i],
                'response': response,
                'topic': topics[i],
                'subtopic': subtopics[i]
            })
        else:
            processed_outputs.append({
                'prompt': orig_que[i],
                'response': response
            })
    return processed_outputs

def save_results(save_name, outputs):
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face model on a dataset.")
    parser.add_argument('--model', help='Hugging Face model path', type=str, required=True)
    parser.add_argument('--save_path', help='Path where the model results will be saved', type=str, default='./generate_results')
    parser.add_argument('--num_samples', help='Number of samples to test from the dataset', type=int, default=-1)
    parser.add_argument('--dataset', help='Path to the dataset (JSON)', type=str, required=True)
    parser.add_argument('--batch_size', help='Batch size for generation', type=int, default=1)
    parser.add_argument('--llama2', action='store_true', help='Use Llama2 model')
    args = parser.parse_args()

    print("\n\nConfiguration")
    print("*" * 10)

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print("*" * 10 + "\n\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

    prompt_que, orig_que, topics, subtopics = load_dataset(args.dataset, args.num_samples, tokenizer)

    os.makedirs(args.save_path, exist_ok=True)

    save_name = os.path.join(args.save_path, f"{os.path.basename(args.dataset).replace('.json', '')}_{os.path.basename(args.model)}.json")
    if args.llama2:
        outputs = generate_responses_llama2(prompt_que, tokenizer, model, args.batch_size)
    else:
        outputs = generate_responses(prompt_que, tokenizer, model, args.batch_size)
    processed_outputs = process_responses(outputs, orig_que, topics, subtopics, args.dataset)

    save_results(save_name, processed_outputs)

    print(f"\nCompleted, please check {save_name}")

if __name__ == "__main__":
    main()
