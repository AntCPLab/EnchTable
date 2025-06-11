import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from tqdm import tqdm
import os

# Assuming SelfReminder is defined somewhere in your code or imported
from utils.self_reminder import SelfReminder

class SaladDataset:
    def __init__(self, input_file):
        self.input_file = input_file
        self.data = self._load_data()
    
    def _load_data(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        return data
    
    def get_questions(self):
        return [entry['question'] for entry in self.data]
    
    def get_batched_questions(self, batch_size):
        questions = self.get_questions()
        for i in range(0, len(questions), batch_size):
            yield questions[i:i + batch_size]
    
    def save_results(self, results, output_path):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using LLaMA3.")
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
        "--batch_size", type=int, default=4, help="Batch size for generating responses"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()

def simplify_model_id(model_id):
    if '/' in model_id:
        # Remote model, take the last part
        simplified_id = model_id.split('/')[-1]
    else:
        # Local model, take the last directory or filename
        simplified_id = os.path.basename(model_id)
    return simplified_id

def generate_response_batch(questions, self_reminder, max_new_tokens=256, do_sample=False, top_p=None):
    responses = []
    lengths = []
    for question in questions:
        response, length = self_reminder.self_reminder(
            question,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p
        )
        responses.append(response)
        lengths.append(length)
    return responses, lengths

def process_dataset(dataset, self_reminder, batch_size, generation_method='sample', **kwargs):
    results = []
    total_questions = len(dataset.data)
    exception_count = 0
    total_batches = (total_questions + batch_size - 1) // batch_size
    with tqdm(
        dataset.get_batched_questions(batch_size),
        desc="Processing batches",
        unit="batch",
        total=total_batches,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    ) as t:
        for batch_questions in t:
            try:
                do_sample = generation_method == 'sample'
                batch_responses, batch_lengths = generate_response_batch(
                    batch_questions, self_reminder, do_sample=do_sample, top_p=kwargs.get('top_p')
                )
                
                for idx, (qid, source, label, question, response) in enumerate(zip(
                    [entry.get('qid', None) for entry in dataset.data],
                    [entry.get('source', None) for entry in dataset.data],
                    [entry.get('1-category', None) for entry in dataset.data],
                    batch_questions,
                    batch_responses
                )):
                    result_entry = {
                        'qid': qid,
                        'source': source,
                        'question': question,
                        'answer': response,
                        '1-category': label  # Assuming '1-category' is part of the dataset
                    }
                    results.append(result_entry)
            except Exception as e:
                logging.error(f"Exception occurred while processing batch: {batch_questions}. Error: {e}")
                for qid, source, label, question in zip(
                    [entry.get('qid', None) for entry in dataset.data],
                    [entry.get('source', None) for entry in dataset.data],
                    [entry.get('1-category', None) for entry in dataset.data],
                    batch_questions
                ):
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

def main():
    args = parse_args()
    
    # Suppress specific warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Set up logging to file
    logging.basicConfig(filename='generate_responses.log', level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the dataset
    dataset = SaladDataset(args.input_file)
    
    # Initialize the SelfReminder class
    self_reminder = SelfReminder(
        base_model_name=args.model_id,
        verbose=args.verbose
    )
    
    results, stats = process_dataset(dataset, self_reminder, args.batch_size, args.generation_method, 
                                    top_p=args.top_p)
    
    simplified_model_id = simplify_model_id(args.model_id)
    output_path = f"./test_salad/llama3/generate/selfreminder/salad_{simplified_model_id}_responses.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.save_results(results, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
