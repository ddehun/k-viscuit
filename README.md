# K-Viscuit :cookie:: Multi-Choice VQA Dataset for Korean Culture

This repository presents the K-Viscuit :cookie: dataset, a Multi-Choice Visual Question Answering (VQA) dataset designed to evaluate Vision-Language Models (VLMs) on Korean culture. 
This dataset is part of the research presented in our paper: [Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration, arXiv 2024 June](https://arxiv.org/abs/2406.16469).
The dataset was created through a Human-VLM collaboration, and examples of the data are as follows.

![Selected examples of K-Viscuit dataset](https://github.com/ddehun/k-viscuit/blob/main/examples.png?raw=true)


## Dataset Availability
The dataset is available both in [this repository](https://github.com/ddehun/k-viscuit/blob/main/dataset/) and [HuggingFace Datasets](https://huggingface.co/datasets/ddehun/k-viscuit).

## Quickstart
To evaluate the [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/tree/main) model on our dataset, please refer to the `run_vqa.py` script provided in this repository. 


## BibTex
For more details about our dataset, please refer to our paper!

```
@article{baek2024evaluating,
  title={Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration},
  author={Baek, Yujin and Park, ChaeHun and Kim, Jaeseok and Heo, Yu-Jung and Chang, Du-Seong and Choo, Jaegul},
  journal={arXiv preprint arXiv:2406.16469},
  year={2024}
}
```
