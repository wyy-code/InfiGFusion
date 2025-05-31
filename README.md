<h1 align="center">
InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion
</h1>

<h4 align="center">

[![Arxiv](https://img.shields.io/badge/Arxiv-D14836?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.13893) 
[![HuggingFace Paper](https://img.shields.io/badge/HuggingFace%20Paper-FF9900?style=for-the-badge&logo=huggingface&logoColor=white)](https://arxiv.org/pdf/2505.13893)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace%20Model-FF9900?style=for-the-badge&logo=huggingface&logoColor=white)](https://arxiv.org/pdf/2505.13893)
  
</h4>

**InfiGFusion** is the first structure-aware fusion framework for large language models that models semantic dependencies among logits using feature-level graphs. We introduce a novel Graph-on-Logits Distillation (GLD) loss that captures cross-dimension interactions via co-activation graphs and aligns them using an efficient, provable approximation of Gromov-Wasserstein distance (reducing complexity from O(n^4) to O(nlogn)). Our released **InfiGFusion-14B** model consistently shows better performance, achieving +35.6 on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and complex logic inference.

## 🤔 Why Model Fusion for LLM?


https://github.com/user-attachments/assets/3f32c836-b5ea-474a-a9d1-53ddf1c259b3



## 📣 News
The ckpt model, InfiGFusion-14B, will be released on Huggingface after acceptance.

## 🎨 Overview
![InfiGFusion_framework](assets/framework.png)

## 🎯 Results
![InfiGFusion](assets/inference.png)
