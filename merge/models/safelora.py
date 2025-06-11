import os
import torch
from transformers import AutoModelForCausalLM

import copy

import numpy
import torch
from transformers import AutoModelForCausalLM

from dataclasses import dataclass, field


def safelora_merge(task_model, taske_pre_model, safety_model, safety_pre_model, select_layers_type='threshold', threshold=0.5, mode='full'):
    safe_lora_config = SafeLoRAConfig()
    safe_lora_config.select_layers_type     = select_layers_type
    safe_lora_config.threshold              = threshold
    safe_lora_config.mode                   = mode

    safe_lora_runner = SafeLoRA(ft_model=task_model, pt_model=taske_pre_model, unaligned_model=safety_pre_model, base_model=safety_model, config=safe_lora_config)

    ft_model = safe_lora_runner.model
    return ft_model

@dataclass
class SafeLoRAConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    unaligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )


    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cpu",
        metadata = {"help": "Devices are used in SafeLoRA. (gpu or cpu)"}

    )

    mode: str = field(
        default="full",
    )
    

    # def __post_init__(self):
    #     if self.base_model_path is None:
    #         raise ValueError("base_model_path cannot be None.")
    #     if self.aligned_model_path is None:
    #         raise ValueError("aligned_model_path cannot be None.")

class SafeLoRA:
    def __init__(self, ft_model:torch.nn.Module, pt_model:torch.nn.Module, unaligned_model:torch.nn.Module, base_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        super().__init__()
        self.ft_model = ft_model
        self.pt_model = pt_model
        self.base_model = base_model
        self.unaligned_model = unaligned_model
        self.config = config
        
        # self.peft_config = ft_model.peft_config["default"]

        if self.config.mode == "lora":
            self.proj_modules = ["q_proj", "v_proj"]
        else:
            self.proj_modules = "full"

        # proj_modules = "all"

        self.model_ori = copy.deepcopy(ft_model)
        project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)
        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            thrs = numpy.sort(cos)[:self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        v = {}
        # proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (self.unaligned_model.named_parameters(), self.base_model.named_parameters()):
            if self.proj_modules == "full" or any(module in a_name for module in self.proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."

                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                

                if len(vec.shape) == 1:
                    vec = vec.unsqueeze(dim = 1)
                    vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                    # vec
                else:
                    vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v[a_name] = (vec).detach().cpu()
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        i = 0
        dis = []
        skip_count = 0
        cos_total = []
        for (name, param),(base_name, base_param) in zip(self.ft_model.named_parameters(), self.pt_model.named_parameters()):
            try:
                if self.proj_modules == "all" or any(module in name for module in self.proj_modules):
                    delta_W = param.data - base_param.data
                    # idx += 1
                    P = v[name].to(param.device)

                    if len(delta_W.shape) == 1:
                        # delta_W = delta_W.unsqueeze(dim = 1)
                        proj_W = torch.mm(P, delta_W.unsqueeze(dim = 1)).squeeze()
                    else:
                        proj_W = torch.mm(P, delta_W)
                    cos = numpy.round(torch.nn.functional.cosine_similarity(proj_W.reshape(1,-1), delta_W.reshape(1,-1)).item(),5)
                    cos_total.append(cos)
                    if cos <=  thrs_cos:
                        i+=1
                        param.data =  base_param.data + proj_W
                        print("safe project layer: {}".format(name))
                    else:
                        # param.data = param_ori
                        skip_count += 1
                        pass
            except:
                continue
                # dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))

                # dis.append(dist.item())

        if show_info:
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
            print("skip_count: {}".format(skip_count))
        return self.ft_model, cos_total