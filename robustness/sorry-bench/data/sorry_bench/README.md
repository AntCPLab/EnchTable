---
license: other
license_name: sorry-bench
license_link: LICENSE
task_categories:
- text-generation
- question-answering
- text2text-generation
language:
- en
- zh
- fr
- ml
- mr
- ta
tags:
- croissant
- safety
extra_gated_fields:
  Name: text
  Affiliation: text
  Email (affiliation email if possible): text
  Country: text
  Purpose: text
size_categories:
- 1K<n<10K
---

<style>
  button {
    /* margin: calc(20vw / 100); */
    margin: 0.5em;
    padding-left:  calc(40vw / 100);
    padding-right:  calc(40vw / 100);
    padding-bottom: calc(0vw / 100);
    text-align: center;
    font-size: 12px;
    height: 25px;
    /* padding-left:  calc(40vw / 100);
    padding-right:  calc(40vw / 100);
    padding-bottom: calc(0vw / 100);
    text-align: center;
    font-size: calc(60vw / 100);
    height: calc(120vw / 100); */
    transition: 0.5s;
    background-size: 200% auto;
    color: white;
    border-radius: calc(60vw / 100);
    display: inline;
    /* border: 2px solid black; */
    font-weight: 500;
    box-shadow: 0px 0px 14px -7px #f09819;
    background-image: linear-gradient(45deg, #FF512F 0%, #F09819 51%, #FF512F 100%);
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
  }

  button:hover {
    background-position: right center;
    /* change the direction of the change here */
    color: #fff;
    text-decoration: none;
  }

  button:active {
    transform: scale(0.95);
  }
</style>

# Dataset Card for SORRY-Bench Dataset (2025/03)

<!-- 📑[**SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors**](https://sorry-bench.github.io) -->

<a href="https://sorry-bench.github.io" style="text-decoration:none">
  <button>🏠Website </button>
</a>
<a href="http://arxiv.org/abs/2406.14598" style="text-decoration:none">
  <button>📑Paper </button>
</a>
<a href="https://huggingface.co/datasets/sorry-bench/sorry-bench-202503" style="text-decoration:none">
  <button>📚Dataset </button>
</a>
<a href="https://github.com/SORRY-Bench/SORRY-Bench" style="text-decoration:none">
  <button>💻Github </button>
</a>
<a href="https://huggingface.co/datasets/sorry-bench/sorry-bench-human-judgment-202503" style="text-decoration:none">
  <button>🧑‍⚖️Human Judgment Dataset </button>
</a>
<a href="https://huggingface.co/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406" style="text-decoration:none">
  <button>🤖Judge LLM </button>
</a>


**🪧UPDATE: In this iteration, we removed the category "Impersonation" due to its ambiguous definition, and that most models more or less fulfill such requests.**

This dataset contains **9.2K potentially unsafe instructions**, intended to be used for LLM safety refusal evaluation.
Particularly, our *base* dataset consists of **440 unsafe instructions** in total, spanning across 44 finegrained safety categories (10 data points per category).
The dataset we present here *equally* captures risks from all safety categories in our taxonomy, whereas prior safety datasets are usually *imbalanced*.

In addition, we paraphrase the base dataset via 20 linguistic mutations (e.g., misspellings, persuasion, translation to non-English languages) that may be easily adopted by real-world users, obtaining 440 * 20 = **8.8K additional unsafe instructions**.


## Dataset Categories (Taxonomy)

This dataset is collected upon our introduced fine-grained **44-class safety taxonomy** (shown below), covering extensive risky topics across **4 high-level domains** (*Hate Speech Generation*, *Assistance with Crimes or Torts*, *Potentially Inappropriate Topics*, and *Potentially Unqualified Advice*).

💡**Note: you can customize your own safety taxonomy to evaluate.** Feel free to selectively engage with categories of particular concerns (the first 24 categories, which are usually considered more prominent), and disregard those deemed permissible (e.g., "25: Advice on Adult Content" or "42: Legal Consulting Advice").

<img src="assets/sorry-bench-taxonomy-10.png" style="width: 80%;" />


## Dataset Structure

**Base Dataset:**
- [question.jsonl](question.jsonl)

  (CSV format: [sorry_bench_202503.csv](sorry_bench_202503.csv))

**Paraphrased Datasets according to 20 Linguistic Mutations:**

- [question_ascii.jsonl](question_ascii.jsonl)
- [question_authority_endorsement.jsonl](question_authority_endorsement.jsonl)
- [question_evidence-based_persuasion.jsonl](question_evidence-based_persuasion.jsonl)
- [question_logical_appeal.jsonl](question_logical_appeal.jsonl)
- [question_misspellings.jsonl](question_misspellings.jsonl)
- [question_question.jsonl](question_question.jsonl)
- [question_slang.jsonl](question_slang.jsonl)
- [question_translate-fr.jsonl](question_translate-fr.jsonl)
- [question_translate-mr.jsonl](question_translate-mr.jsonl)
- [question_translate-zh-cn.jsonl](question_translate-zh-cn.jsonl)
- [question_atbash.jsonl](question_atbash.jsonl)
- [question_caesar.jsonl](question_caesar.jsonl)
- [question_expert_endorsement.jsonl](question_expert_endorsement.jsonl)
- [question_misrepresentation.jsonl](question_misrepresentation.jsonl)
- [question_morse.jsonl](question_morse.jsonl)
- [question_role_play.jsonl](question_role_play.jsonl)
- [question_technical_terms.jsonl](question_technical_terms.jsonl)
- [question_translate-ml.jsonl](question_translate-ml.jsonl)
- [question_translate-ta.jsonl](question_translate-ta.jsonl)
- [question_uncommon_dialects.jsonl](question_uncommon_dialects.jsonl)

**List of 44 Safety Categories:**

- [meta_info.py](meta_info.py)


## Uses

This dataset contains numerous potentially unsafe instructions across 44 finegrained safety categories, intended to be used for (large) language model safety evaluation.
Specifically:

- **LLM developers** can use this dataset to systematically check model *safety refusal behaviors* on each of our 44 safety categories (or on a subset of these categories, according to their own safety policies).
  This can be done conveniently via our benchmark pipeline (see our [Github repo](https://github.com/SORRY-Bench/SORRY-Bench)).
  Further, we recommend model developers thoroughly evaluate additional safety risks associated with the *20 linguistic mutations* we have integrated in our paraphrased datasets, which may be adopted by average-case real-world users.
- **Jailbreak researchers** can benchmark the effectiveness of different jailbreaking attacks and defenses on our dataset, in a more granular and comprehensive manner.



## Dataset Creation

We manually curate numerous novel unsafe instructions, which comprise the majority part of our base dataset.
Meanwhile, for a small portion of our dataset, we also reuse existing data points (modified and rewritten) from 10 prior safety datasets ([HarmfulQ](https://github.com/SALT-NLP/chain-of-thought-bias), [AdvBench](https://github.com/llm-attacks/llm-attacks), [Do-Anything-Now](https://github.com/verazuo/jailbreak_llms), [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM), [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI), [SimpleSafetyTest](https://github.com/bertiev/SimpleSafetyTests), [FFT](https://github.com/cuishiyao96/FFT), [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat), [HarmBench](https://github.com/centerforaisafety/HarmBench), [StrongREJECT](https://github.com/alexandrasouly/strongreject)).
Refer to our 📑[SORRY-Bench paper](https://arxiv.org/abs/2406.14598) for details.


## SORRY-Bench Dataset License Agreement

This Agreement contains the terms and conditions that govern your access and use of the SORRY-Bench Dataset (as defined above). You may not use the SORRY-Bench Dataset if you do not accept this Agreement. By clicking to accept, accessing the SORRY-Bench Dataset, or both, you hereby agree to the terms of the Agreement. If you are agreeing to be bound by the Agreement on behalf of your employer or another entity, you represent and warrant that you have full legal authority to bind your employer or such entity to this Agreement. If you do not have the requisite authority, you may not accept the Agreement or access the SORRY-Bench Dataset on behalf of your employer or another entity.

* Safety and Moderation: **This dataset contains unsafe conversations or prompts that may be perceived as offensive or unsettling.** Users may not use this dataset for training machine learning models for any harmful purpose. The dataset may not be used to generate content in violation of any law. These prompts should not be used as inputs to models that can generate modalities outside of text (including, but not limited to, images, audio, video, or 3D models)
* Non-Endorsement: The views and opinions depicted in this dataset **do not reflect** the perspectives of the researchers or affiliated institutions engaged in the data collection process.
* Legal Compliance: You are mandated to use it in adherence with all pertinent laws and regulations.
* Model Specific Terms: When leveraging direct outputs of a specific model, users must adhere to its **corresponding terms of use and relevant legal standards**.
* Non-Identification: You **must not** attempt to identify the identities of individuals or infer any sensitive personal data encompassed in this dataset.
* Prohibited Transfers: You **should not** distribute, copy, disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third party.
* Right to Request Deletion: At any time, we may require you to delete all copies of this instruction dataset (in whole or in part) in your possession and control. You will promptly comply with any and all such requests. Upon our request, you shall provide us with written confirmation of your compliance with such requirement.
* Termination: We may, at any time, for any reason or for no reason, terminate this Agreement, effective immediately upon notice to you. Upon termination, the license granted to you hereunder will immediately terminate, and you will immediately stop using the SORRY-Bench Dataset and destroy all copies of the SORRY-Bench Dataset and related materials in your possession or control.
* Limitation of Liability: IN NO EVENT WILL WE BE LIABLE FOR ANY CONSEQUENTIAL, INCIDENTAL, EXEMPLARY, PUNITIVE, SPECIAL, OR INDIRECT DAMAGES (INCLUDING DAMAGES FOR LOSS OF PROFITS, BUSINESS INTERRUPTION, OR LOSS OF INFORMATION) ARISING OUT OF OR RELATING TO THIS AGREEMENT OR ITS SUBJECT MATTER, EVEN IF WE HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Subject to your compliance with the terms and conditions of this Agreement, we grant to you, a limited, non-exclusive, non-transferable, non-sublicensable license to use the SORRY-Bench Dataset, including the conversation data and annotations, to research, and evaluate software, algorithms, machine learning models, techniques, and technologies for both research and commercial purposes.


## Citation

**BibTeX:**

```
@inproceedings{
xie2025sorrybench,
title={{SORRY}-Bench: Systematically Evaluating Large Language Model Safety Refusal},
author={Tinghao Xie and Xiangyu Qi and Yi Zeng and Yangsibo Huang and Udari Madhushani Sehwag and Kaixuan Huang and Luxi He and Boyi Wei and Dacheng Li and Ying Sheng and Ruoxi Jia and Bo Li and Kai Li and Danqi Chen and Peter Henderson and Prateek Mittal},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=YfKNaRktan}
}
```

## Dataset Card Contact

Tinghao Xie (thx@princeton.edu)