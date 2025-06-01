<h1 align="center">
InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion
</h1>

<h4 align="center">

[![Arxiv](https://img.shields.io/badge/Arxiv-D14836?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.13893) 
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FF9900?style=for-the-badge&logo=huggingface&logoColor=white)](https://arxiv.org/pdf/2505.13893)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace%20Model-FF9900?style=for-the-badge&logo=huggingface&logoColor=white)](https://arxiv.org/pdf/2505.13893)
  
</h4>

**InfiGFusion** is the first structure-aware fusion framework for large language models that models semantic dependencies among logits using feature-level graphs. We introduce a novel Graph-on-Logits Distillation (GLD) loss that captures cross-dimension interactions via co-activation graphs and aligns them using an efficient, provable approximation of Gromov-Wasserstein distance (reducing complexity from O(n^4) to O(nlogn)). Our released **InfiGFusion-14B** model consistently shows better performance, achieving +35.6 on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and complex logic inference.

## 📣 News
😁 The ckpt model, InfiGFusion-14B, has been released on Huggingface! ! !

## 📕 Model Summary 

|                         |                                                                               |     
|-------------------------|-------------------------------------------------------------------------------|
| **Developers**          | Reallm-Labs                                                            |
| **Description**         | InfiGFusion is an open fusion model series designed to fuse multiple domain LLMs into a single LLM. It excels in multi-step and relational inference, enabling robust performance across complex reasoning tasks.|
| **Architecture**        | 14B parameters, dense decoder-only Transformer model                          |
| **Inputs**              | Text, best suited for prompts in the chat format                              |
| **Max Context length**  | 16K tokens                                                                    |
| **GPUs**                | 48 H100-80G                                                                 |
| **Fusing input length**      | 4K tokens                                                           |
| **Fusing time**       | 192 hours                                                                       |
| **Fusing data**       | 520M tokens                                                                   |
| **Outputs**             | Generated text in response to input                                           |
| **Status**              | Static model trained on an offline dataset                                    |
| **License**             | MIT                                                                         |

## 🎨 Overview
![InfiGFusion_framework](assets/framework.png)

## 🎯 Results
![InfiGFusion](assets/inference.png)
