# Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models [ACL 2024]
[arXiv](https://arxiv.org/pdf/2307.01379)

**Authors:** Jinhao Duan, Hao Cheng, Shiqi Wang, Chenan Wang, Alex Zavalny, Renjing Xu, Bhavya Kailkhura, Kaidi Xu

The proposed Shifting-Attention-to-Relevance (SAR) is implemented in this codebase. 

## Updates

**[8/2024]** 🎉🎉 Glad to know that SAR is ranked **1st** among 28 LLM uncertainty quantification methods in [LM-Polygraph](https://github.com/IINemo/lm-polygraph?tab=readme-ov-file). Please also check their implementation and [paper](https://arxiv.org/pdf/2406.15627).

## Environments

Please config environment by 

```pip install -r requirements.txt```

## Data Preparing
```shell
cd src
sh parse_datasets.sh
```
It will automatically parse CoQA, Trivia QA, and SciQ datasets.

## Uncertainty Estimation for Open-source LLMs 
#### for the CoQA dataset
```shell
sh scripts/coqa/ue_pipeline_opt-2.7b.sh

sh scripts/coqa/ue_pipeline_opt-6.7b.sh

sh scripts/coqa/ue_pipeline_opt-13b.sh

sh scripts/coqa/ue_pipeline_opt-30b.sh

sh scripts/coqa/ue_pipeline_llama-7b.sh

sh scripts/coqa/ue_pipeline_llama-13b.sh
````

#### for the SciQ dataset:
```shell
sh scripts/sciq/ue_pipeline_opt-2.7b.sh

sh scripts/sciq/ue_pipeline_opt-6.7b.sh

sh scripts/sciq/ue_pipeline_opt-13b.sh

sh scripts/sciq/ue_pipeline_opt-30b.sh

sh scripts/sciq/ue_pipeline_llama-7b.sh

sh scripts/sciq/ue_pipeline_llama-13b.sh
```

#### for the Trivia QA dataset:
```shell
sh scripts/trivia_qa/ue_pipeline_llama-7b.sh

sh scripts/trivia_qa/ue_pipeline_llama-13b.sh
```

## Reference
Please cite our paper if you feel this is helpful:
```shell
@inproceedings{duan2024shifting,
  title={Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models},
  author={Duan, Jinhao and Cheng, Hao and Wang, Shiqi and Zavalny, Alex and Wang, Chenan and Xu, Renjing and Kailkhura, Bhavya and Xu, Kaidi},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5050--5063},
  year={2024}
}
```

## Acknowledgement
This codebase is build upon [Semantic Entropy (SE)](https://github.com/lorenzkuhn/semantic_uncertainty). Thanks for their excellent contribution!
