
# EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2511.09880-b31b1b.svg)](https://arxiv.org/abs/2511.09880)

</div>

## Overview

EnchTable is a unified framework for transferring safety alignment to fine-tuned large language models without extensive retraining. It combines NTK-based safety vector distillation to extract safety knowledge, and an interference-aware merging strategy to preserve both safety and utility. Evaluated across diverse models and tasks, EnchTable effectively mitigates safety degradation during fine-tuning, maintains high task performance, and shows strong robustness against jailbreak attacks.


---

## 🛠️ Preparation

> **Model weights (both harmful and realigned) will be released on Hugging Face for easy access and reproducibility.**

You can build the required environment by running:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The entire workflow consists of two main stages:

1. **Safety Distillation**
2. **Merge**

### 1. Safety Distillation
For safety distillation, we modify the codebase of **LLaMA Factory** to implement NTK-constrained fine-tuning. This step aims to extract safety vector from surrogate LLM.

> ⚠️ Make sure you have properly configured LLaMA Factory before proceeding.

To run the default distillation configuration, simply execute:

```bash
bash ./safety_distillation/LLaMA-Factory/train.sh
```

After training, the harmful surrogate model will be saved into `./safety_distillation/LLaMA-Factory/saves/llama3-8b-beavertail_harmful/attention/sft_ntk_linear_e4`

### 2. Merge
We implement both baseline merging strategies and our proposed **interference-aware merging** method in `./merge/merge.py`.

To run the default merging configuration, simply execute:

```bash
bash ./run.sh
```

This script runs the merging process with default hyperparameters (e.g., $\beta = 0.1$, $\gamma = 0.5$). You can customize these values via command-line arguments or by editing the script directly. The merged model will be saved into `./merge/merged_models/Code-Llama-3-8B_aligned`.

---

## 📊 Evaluation

We evaluate the merged model using multiple benchmarks across different domains:

- **Code generation**: Evaluated using [EvalPlus](https://github.com/evalplus/evalplus)
- **Math reasoning**: Evaluated using [math-eval-harness](https://github.com/ZubinGou/math-evaluation-harness)
- **Medical/general safety**: Evaluated using [lm-harness](https://github.com/EleutherAI/lm-evaluation-harness)

> ⚠️ **Before running the evaluations, please install the required evaluation packages for each benchmark.** Below are the installation instructions:

```bash
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

git clone https://github.com/ZubinGou/math-evaluation-harness.git
cd math-evaluation-harness
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
---

### 🔐 Safety Evaluation

To evaluate the safety of the merged model, follow these steps:

#### Generate responses:
```bash
bash eval/scripts/generate_other.sh   # Generates responses for general safety evaluation
bash eval/scripts/generate_salad.sh   # Generates responses for SALADBench safety evaluation
```

#### Judge safety:
```bash
bash eval/scripts/judge_other.sh      # Judges general safety
bash eval/scripts/judge_salad.sh      # Judges SALADBench safety
```

These scripts will output metrics such as Unsafe Rate, which reflect how well the merged model adheres to safety guidelines.

---

### 🧪 Utility Evaluation

To evaluate the utility (task performance) of the merged model:

#### Code generation:
```bash
bash eval/scripts/evalplus.sh         # Evaluates code generation capability
```

#### Math reasoning:
```bash
bash eval/math-evaluation-harness/eval.sh         # Evaluates math reasoning capability
```

#### Medical/general reasoning:
```bash
bash eval/scripts/evalharness.sh      # Evaluates reasoning capability on medical tasks
```

These scripts report standard metrics: accuracy.

---

## 🔍 Robustness Evaluation

To comprehensively assess the safety robustness of models under adversarial prompting, we employ two widely used benchmarks: **SorryBench** and **AISafetyLab**. These tools simulate real-world "jailbreak" attacks, allowing us to evaluate the effectiveness of our safety alignment framework in extreme scenarios.

### SorryBench

#### Usage:

```bash
cd robustness/sorry-bench
bash generate_answer.sh  # Generate model responses
bash judge.sh            # Use built-in judge model to detect jailbreak success
```

> ⚠️ Note: You need to download the judge model provided by SorryBench and ensure its path is correctly configured.


### AISafetyLab


#### Installation:

```bash
git clone https://github.com/thu-coai/AISafetyLab.git
cd AISafetyLab
pip install -e .
```

#### Example Attack Configurations

##### DRA Attack Configuration:

```yaml
attack_data_path: 'thu-coai/AISafetyLab_Datasets/harmbench_standard'
target_model_path: "YOUR_PATH"
target_model_name: "llama3"
demo_num: 1
lang: en
evaluator_type: "pattern"
evaluator_model_path: "meta-llama/Llama-Guard-3-8B"
detoxify_model_path: "./detoxify_model/toxic_original-c1212f89.ckpt"
detoxify_config_path: "./model_bert"
res_save_path: './results/dra_Code-Llama-3-8B_harmbench_50.jsonl'
device: "cuda:2"
iters: 20
em_t: 0.7
```

##### ICA Attack Configuration:

```yaml
attack_data_path: "thu-coai/AISafetyLab_Datasets/harmbench_standard" 
target_model_path: "ajibawa-2023/Code-Llama-3-8B"
target_model_name: "llama3" 
demo_num: 1
lang: en
evaluator_type: "pattern" 
evaluator_path: "meta-llama/Llama-Guard-3-8B"
res_save_path: "./results/ica_Code-Llama-3-8B_harmbench_1shot_PatternScore.jsonl"
device: "cuda:0"
```

#### Running an Attack (Example with DRA):

```bash
python run_attack.py --config configs/dra.yaml
```
---

## 📎 Citation

If you find EnchTable useful in your research, please cite our paper:

```bibtex
@article{wu2025enchtable,
  title={EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models},
  author={Wu, Jialin and Li, Kecen and Huang, Zhicong and Li, Xinfeng and Wang, Xiaofeng and Hong, Cheng},
  journal={arXiv preprint arXiv:2511.09880},
  year={2025}
}
```

---

## 📬 Contact

For questions, collaboration, or feedback, feel free to reach out:

📧 [jinlin.wjl@antgroup.com](mailto:jinlin.wjl@antgroup.com) or [wjlinzju@gmail.com](wjlinzju@gmail.com)

We welcome contributions and discussions!
