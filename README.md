# Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models [ACL 2024]
[arXiv](https://arxiv.org/pdf/2307.01379)

**Authors:** Jinhao Duan, Hao Cheng, Shiqi Wang, Chenan Wang, Alex Zavalny, Renjing Xu, Bhavya Kailkhura, Kaidi Xu

The proposed Shifting-Attention-to-Relevance (SAR) is implemented in this codebase. 

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
@article{duan2023shifting,
  title={Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models},
  author={Duan, Jinhao and Cheng, Hao and Wang, Shiqi and Wang, Chenan and Zavalny, Alex and Xu, Renjing and Kailkhura, Bhavya and Xu, Kaidi},
  journal={arXiv preprint arXiv:2307.01379},
  year={2023}
}
```

## Acknowledgement
This codebase is build upon [Semantic Entropy (SE)](https://github.com/lorenzkuhn/semantic_uncertainty). Thanks for their excellent contribution!
