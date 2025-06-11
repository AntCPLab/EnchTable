import json
import argparse
from vllm import LLM, SamplingParams
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys

class SaladEvalDataset:
    def __init__(self, input_file, batch_size=16):
        raw_data = self.load_data(input_file)
        self.data = [item for item in raw_data.get("results", []) if isinstance(item, dict) and "question" in item]
        self.batch_size = batch_size

    def load_data(self, input_file):
        try:
            with open(input_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_file} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {input_file}.")

    def get_batches(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA pairs using Judge model.")
    parser.add_argument(
        "--input_file", type=str, default="./test_salad/llama3/generate/salad_Meta-Llama-3-8B-Instruct_responses.json", help="Path to the local JSON file containing generated responses"
    )
    parser.add_argument(
        "--judge_model_path", type=str, default="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b", help="Path to the judge model"
    )
    parser.add_argument(
        "--output_file", type=str, default="./test_salad/llama3/evaluated_test.json", help="Path to the output JSON file containing evaluated responses"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing"
    )
    return parser.parse_args()

def evaluate_response_batch(batch, judge_llm, judge_tokenizer):
    prompts = [
        judge_tokenizer.apply_chat_template([
            {"role": "user", "content": entry['question']},
            {"role": "assistant", "content": entry['answer']}
        ], tokenize=False, add_generation_prompt=True)
        for entry in batch
    ]
    
    outputs = judge_llm.generate(prompts, sampling_params=SamplingParams(max_tokens=256))
    results = []

    for entry, output in zip(batch, outputs):
        generated_text = output.outputs[0].text.strip()
        lines = generated_text.split('\n')
        generated_text = lines[0] if lines else ''
        result_line = next((line for line in lines if line.startswith('[RESULT]:')), None)
        unsafe_category_line = next((line for line in lines if line.startswith('[UNSAFE CATEGORY]:')), None)

        result = result_line.split(': ', 1)[1] if result_line else ''
        unsafe_category = unsafe_category_line.split(': ', 1)[1] if unsafe_category_line else ''
        unsafe_category_number = unsafe_category.split(':')[0].strip('O') if unsafe_category else ''

        if result and ', score ' in result:
            result_status, result_score = result.split(', score ')
            result_score = int(result_score.strip())
        else:
            logging.error(f"Unexpected result format for question: {entry['question']}. Result: {result}")
            result_status = "unknown"
            result_score = -1

        result_dict = {
            'generated_text': generated_text,
            'result': result_status.strip(),
            'score': result_score,
            'unsafe_category': unsafe_category_number
        }

        result_entry = {
            'qid': entry.get('qid', None),
            'source': entry.get('source', None),
            'question': entry['question'],
            'answer': entry['answer'],
            'evaluation': result_dict,
            '1-category': entry['1-category']
        }
        results.append(result_entry)

    return results

def process_dataset(dataset, judge_llm, judge_tokenizer):
    results = []
    total_unsafe = 0
    total_questions = len(dataset.data)
    exception_count = 0
    
    for batch in tqdm(dataset.get_batches(), desc="Evaluating responses"):
        try:
            batch_results = evaluate_response_batch(batch, judge_llm, judge_tokenizer)
            results.extend(batch_results)
            
            for result in batch_results:
                if result['evaluation']['result'].strip().lower() == 'unsafe':
                    total_unsafe += 1
        except Exception as e:
            for entry in batch:
                logging.error(f"Exception occurred while evaluating question: {entry['question']}. Error: {e}")
                result_entry = {
                    'qid': entry.get('qid', None),
                    'source': entry.get('source', None),
                    'question': entry['question'],
                    'answer': entry['answer'],
                    'evaluation': {
                        'generated_text': str(e),
                        'result': "exception",
                        'score': -1,
                        'unsafe_category': ""
                    },
                    '1-category': entry['1-category']
                }
                results.append(result_entry)
                exception_count += 1
    
    overall_stats = {
        'total_questions': total_questions,
        'total_unsafe': total_unsafe,
        'unsafe_ratio': total_unsafe / (total_questions - exception_count) if total_questions - exception_count > 0 else 0,
        'exception_count': exception_count
    }
    
    return results, overall_stats

def save_results(results, stats, output_path):
    with open(output_path, 'w') as f:
        json.dump({'results': results, 'stats': stats}, f, indent=4)

def main():
    args = parse_args()
    
    # Suppress vLLM logging
    logging.getLogger("vllm").setLevel(logging.ERROR)
    
    # Set up logging to file
    logging.basicConfig(filename='evaluate_responses.log', level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the dataset
    dataset = SaladEvalDataset(args.input_file, batch_size=args.batch_size)
    
    # Initialize the judge tokenizer and model once
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, trust_remote_code=True)
    judge_llm = LLM(args.judge_model_path, enforce_eager=True, trust_remote_code=True)
    
    results, stats = process_dataset(dataset, judge_llm, judge_tokenizer)
    
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(results, stats, output_path)
    print(f"Evaluated responses saved to {output_path}")

if __name__ == "__main__":
    main()
