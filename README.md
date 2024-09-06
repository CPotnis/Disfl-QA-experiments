# Optimizing Llama 3.1 for Disfluent QA with Prompt Engineering and Fine-Tuning Techniques

## Overview

This repository features experiments with the state-of-the-art Llama 3.1 language model, employing advanced prompt engineering and Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LORA). The key focus is on developing a question rewrite model for the Disfl QA benchmark dataset to enhance the handling of disfluent or noisy inputs, with the goal of accurate question extraction and improved intent prediction.

## Experiments

### Experiment 1: Prompt Engineering (No Fine-Tuning)
- **Objective**: To evaluate the model's baseline capabilities using advanced prompt engineering techniques without altering the model's parameters.
- **Model**: Llama 3.1 (Non-fine-tuned)
- **Approach**: Systematic design of prompts to guide the model's output.
- **Dataset**: Full dataset (Disfl-QA).

### Experiment 2: Fine-Tuning with LORA on Full Dataset
- **Objective**: To assess the performance of the model fine-tuned using PEFT (LORA) with a full training dataset.
- **Model**: Llama 3.1 fine-tuned with LORA adapters.
- **Approach**: Training 16 rank-structured LORA adapters for one epoch on the complete dataset.
- **Dataset**: Full dataset (Disfl-QA).

### Experiment 3: Fine-Tuning with LORA on Reduced Dataset
- **Objective**: To evaluate the model's performance with PEFT (LORA) on a reduced dataset (100 rows), comparing the results to the full dataset.
- **Model**: Llama 3.1 fine-tuned with LORA adapters.
- **Approach**: Training 16 rank-structured LORA adapters for 5 epochs on the reduced dataset.
- **Dataset**: Reduced dataset (100 rows).

## Structure

- `Experiment_1_Prompt_Engineering.ipynb`: Notebook for Experiment 1.
- `Experiment_2_LORA_Full_Dataset.ipynb`: Notebook for Experiment 2.
- `Experiment_3_LORA_Reduced_Dataset.ipynb`: Notebook for Experiment 3.

## Requirements

### Packages

Ensure the following packages are installed:

- `unsloth`
- `rouge_score`
- `evaluate`
- `xformers`
- `trl`
- `peft`
- `accelerate`
- `bitsandbytes`
- `triton`

### Hugging Face Token

To access the Llama 3.1 model, you will need to set up a Hugging Face token.

1. Go to your Hugging Face profile and create a new token [here](https://huggingface.co/settings/tokens).
2. Add the token in your notebook or environment as follows:

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

## Results

The following tables compare the performance and infrastructure details across all three experiments.

### Table 1: Model Performance Metrics

| Experiment         | BLEU Score | ROUGE-1 Score | ROUGE-2 Score | ROUGE-L Score | Training Loss |
|--------------------|------------|-------------|---------------|---------------|---------------|
| Prompt Engineering | 0.488      | 0.764         |  0.642          |  0.741          | N/A           |
| LORA Full Dataset  | 0.887        | 0.953         | 0.912          | 0.941           | 0.408600          |
| LORA Reduced Dataset | 0.882      | 0.942         | 0.898           | 0.933           | 0.453500         |

### Table 2: GPU Infrastructure

| Experiment         | GPU Name        | GPU Memory (GB) | Model Size in Memory (GB) |
|--------------------|-----------------|-----------------|---------------------------|
| Prompt Engineering | Nvidia T4       | 14.748 GB.      | 5.9GB                         |
| LORA Full Dataset  | Nvidia A100     | 39.564 GB       | 5.9GB                       |
| LORA Reduced Dataset | Nvidia A100   | 39.564 GB       | 5.9GB                       |

### Table 3: Training Metrics

| Experiment         | Num Rows | Training Time (hrs) | Batch Size | Epochs |
|--------------------|----------|---------------------|------------|--------|
| Prompt Engineering | N/A      | N/A                 | N/A        | N/A    |
| LORA Full Dataset  | 7182 (Full Training Set) | 6 Minutes                 | 128        | 1      |
| LORA Reduced Dataset | 100    |   1 Minute    | 8        | 5      |


LORA fine-tuning delivered the best results, with the Full Dataset approach achieving the highest BLEU and ROUGE scores. Prompt engineering was less effective by comparison.


