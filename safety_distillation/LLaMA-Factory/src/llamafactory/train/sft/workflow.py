# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer



import abc
import os

import torch
import torch.nn as nn
import copy
from functorch import jvp, make_functional_with_buffers
from transformers.modeling_outputs import BaseModelOutputWithPast

class LinearizedModel(nn.Module):

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.original_model = {'original_model': model.to('cpu')}

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp
    
    def recover(self):
        device = self.params[0].device
        i = 0
        model_dict = {}
        for k, _ in self.original_model['original_model'].state_dict().items():
            model_dict[k] = self.params[i].to('cpu')
            i += 1
        self.original_model['original_model'].load_state_dict(model_dict)
        return self.original_model['original_model'].to(device)

class LinearizedLlamaModel(nn.Module):

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, **kwargs: func0(params, self.buffers0, **kwargs)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )
        self.original_model = {'original_model': model.to('cpu')}

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            if p.requires_grad:
                p.requires_grad = True
            else:
                p.requires_grad = False
    def __call__(self, **kwargs) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp, (all_hidden_states, all_self_attns) = jvp(
            lambda param: self.func0(param, **kwargs),
            (tuple(self.params0),),
            (tuple(dparams),),
            has_aux=True,
        )
        hidden_states = out + dp
        # add hidden states from the last decoder layer
        if kwargs['output_hidden_states']:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=kwargs['past_key_values'] if kwargs['use_cache'] else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if kwargs['return_dict'] else output.to_tuple()
    
    def recover(self):
        device = self.params[0].device
        i = 0
        model_dict = {}
        for k, _ in self.original_model['original_model'].state_dict().items():
            model_dict[k] = self.params[i].to('cpu')
            i += 1
        self.original_model['original_model'].load_state_dict(model_dict)
        return self.original_model['original_model'].to(device)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # ntk = [None, 'linear', 'modulelist'][1]
    ntk = [None, 'attn', 'ffn', 'attn_ffn', 'modulelist'][1]

    # print(model.model)
    # print(model.base_model)
    if ntk == 'attn':
        for i in range(len(model.base_model.layers)):
            model.model.layers[i].self_attn.q_proj = LinearizedModel(model.model.layers[i].self_attn.q_proj)
            model.model.layers[i].self_attn.k_proj = LinearizedModel(model.model.layers[i].self_attn.k_proj)
            model.model.layers[i].self_attn.v_proj = LinearizedModel(model.model.layers[i].self_attn.v_proj)
            model.model.layers[i].self_attn.o_proj = LinearizedModel(model.model.layers[i].self_attn.o_proj)
    elif ntk == 'ffn':
        for i in range(finetuning_args.freeze_trainable_layers, len(model.base_model.layers)-finetuning_args.freeze_trainable_layers):
            model.model.layers[i].mlp = LinearizedModel(model.model.layers[i].mlp)
    elif ntk == 'attn_ffn':
        for i in range(finetuning_args.freeze_trainable_layers, len(model.base_model.layers)-finetuning_args.freeze_trainable_layers):
            model.model.layers[i].self_attn.q_proj = LinearizedModel(model.model.layers[i].self_attn.q_proj)
            model.model.layers[i].self_attn.k_proj = LinearizedModel(model.model.layers[i].self_attn.k_proj)
            model.model.layers[i].self_attn.v_proj = LinearizedModel(model.model.layers[i].self_attn.v_proj)
            model.model.layers[i].self_attn.o_proj = LinearizedModel(model.model.layers[i].self_attn.o_proj)
            model.model.layers[i].mlp = LinearizedModel(model.model.layers[i].mlp)
    elif ntk == 'modulelist':
        print(type(model.model))
        model.model = LinearizedLlamaModel(model.model)
        print("********************")
        print(type(model.model))
    # model.base_model.layers[0].self_attn.q_proj = LinearizedModel(model.base_model.layers[0].self_attn.q_proj)

    # kecen ntk
    # model.model.
    # if model_args.use_linearized_model:
    # print("+++++++++++++++",type(model))
    # model = LinearizedModel(model, init_model=model)
    # print("+++++--------",type(model))
    
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])
    if trainer.is_world_process_zero():
        if ntk == 'attn':
            for i in range(len(model.base_model.layers)):
                model.model.layers[i].self_attn.q_proj = model.model.layers[i].self_attn.q_proj.recover()
                model.model.layers[i].self_attn.k_proj = model.model.layers[i].self_attn.k_proj.recover()
                model.model.layers[i].self_attn.v_proj = model.model.layers[i].self_attn.v_proj.recover()
                model.model.layers[i].self_attn.o_proj = model.model.layers[i].self_attn.o_proj.recover()
            model.save_pretrained(trainer.args.output_dir)
        elif ntk == 'ffn':
            for i in range(finetuning_args.freeze_trainable_layers, len(model.base_model.layers)-finetuning_args.freeze_trainable_layers):
                model.model.layers[i].mlp = model.model.layers[i].mlp.recover()
            model.save_pretrained(trainer.args.output_dir)
        elif ntk == 'attn_ffn':
            for i in range(finetuning_args.freeze_trainable_layers, len(model.base_model.layers)-finetuning_args.freeze_trainable_layers):
                model.model.layers[i].self_attn.q_proj = model.model.layers[i].self_attn.q_proj.recover()
                model.model.layers[i].self_attn.k_proj = model.model.layers[i].self_attn.k_proj.recover()
                model.model.layers[i].self_attn.v_proj = model.model.layers[i].self_attn.v_proj.recover()
                model.model.layers[i].self_attn.o_proj = model.model.layers[i].self_attn.o_proj.recover()
                model.model.layers[i].mlp = model.model.layers[i].mlp.recover()
            model.save_pretrained(trainer.args.output_dir)
        elif ntk == 'modulelist':
            model.model = model.model.recover()
            model.save_pretrained(trainer.args.output_dir)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
