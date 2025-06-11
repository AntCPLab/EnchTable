import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

from models.tsvm import tsvm_merge
from models.resta import resta_merge
from models.safelora import safelora_merge
# from models.ties import ties_merge
from models.resta_tsvm import resta_tsvm_merge
# from models.model_stock import stock_merge
# from models.model_breadcrumbs import breadcrumbs_merge


def main(args):
    task_model = AutoModelForCausalLM.from_pretrained(args.task_model)
    tokenizer = AutoTokenizer.from_pretrained(args.task_model)
    taske_pre_model = AutoModelForCausalLM.from_pretrained(args.task_model_pre)
    safety_model = AutoModelForCausalLM.from_pretrained(args.safety_model)
    safety_pre_model = AutoModelForCausalLM.from_pretrained(args.safety_model_pre)
    os.makedirs(args.save_path, exist_ok=True)
    if args.method == 'safelora':
        merged_model = safelora_merge(task_model, taske_pre_model, safety_model, safety_pre_model)
    elif args.method == 'resta':
        merged_model = resta_merge(task_model, taske_pre_model, safety_model, safety_pre_model, adaptive=args.ada, model_name=args.task_model if args.iscore else None)
    # elif args.method == 'model_stock':
    #     merged_model = stock_merge(task_model, taske_pre_model, safety_model, safety_pre_model)
    # elif args.method == 'model_breadcrumbs':
    #     merged_model = breadcrumbs_merge(task_model, taske_pre_model, safety_model, safety_pre_model)
    elif args.method == 'resta_tsvm':
        merged_model = resta_tsvm_merge(task_model, taske_pre_model, safety_model, safety_pre_model, adaptive=args.ada, ada_alpha=args.ada_alpha)
    elif 'tsvm' in args.method:
        merged_model = tsvm_merge(task_model, taske_pre_model, safety_model, safety_pre_model, mode=int(args.method[-1]))
    # elif args.method == 'ties':
    #     merged_model =  ties_merge(task_model, taske_pre_model, safety_model, safety_pre_model)
    else:
        raise NotImplementedError

    merged_model.generation_config.temperature = None
    merged_model.generation_config.top_p = None
    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_model', help='The fine-tuned model , e.g. local model path.', type=str, required=True)
    parser.add_argument('--task_model_pre', help='The base model for alignment, e.g. local model path.', type=str, required=True)
    parser.add_argument('--safety_model', help='The aligned model path.', type=str, required=False, default=None)
    parser.add_argument('--safety_model_pre', help='The unaligned model path to compare against the aligned model.', type=str, required=True)
    parser.add_argument('--save_path', help='Path where the model results will be saved. Default is "evaluate/results".', type=str, required=True, default='evaluate/results')
    parser.add_argument('--method', type=str, required=True, default="resta", choices=['resta', 'tsvm_1', 'safelora', 'ties', 'resta_tsvm', 'model_stock', 'model_breadcrumbs'])
    parser.add_argument('--m0', type=float, default=0.)
    parser.add_argument('--ada', action='store_true')
    parser.add_argument('--ada_alpha', type=float, default=0.1)
    parser.add_argument('--iscore', action='store_true')

    args = parser.parse_args()
    if args.safety_model is None:
        args.safety_model = args.task_model_pre
    main(args)