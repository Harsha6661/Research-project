## A Research Study on Multimodal Hallucinations

A supervised hallucination detection system for text and image modalities, built by adapting the POPE (Polling-based Object Probing Evaluation) framework with TinyBERT. Achieved 98.12% accuracy on the text modality, outperforming the POPE baseline by approximately 10 percentage points.

This work was completed as a group research project for the MSc Data Science and AI programme at the University of Liverpool (Semester 1, 2025).

### Overview

Multimodal Large Language Models (MLLMs) frequently produce hallucinations — fluent but factually inaccurate content inconsistent with the input. While the POPE framework introduced a useful binary probing approach for evaluating object hallucination, it remains a discriminative evaluation tool rather than a trainable detection system.

This project adapts POPE into a fully supervised, scalable detection pipeline that works across text and image modalities, with the goal of producing a simple, reliable, and resource-efficient detector.

### Approach

The system reformulates hallucination detection as a binary classification task. For a given input (text document or image) and a candidate entity, the model predicts whether that entity is genuinely present or hallucinated.

Text pipeline:

- Source: ~11,100 long-form documents, filtered to ~10,000 usable documents
- Entity extraction: capitalized words and frequent nouns, with strict token-boundary matching
- POPE-style QA generation: ~100,000 binary (entity-present / entity-absent) question pairs
- Model: pretrained TinyBERT fine-tuned with a linear classification head on the [CLS] token
- Input format: [CLS] question [SEP] document [SEP], 256-token cap with leading-paragraph truncation
- Training: 5 epochs on consumer hardware (MacBook CPU), class-balanced sampler to address label imbalance

Image pipeline: extends the same POPE-style binary probing approach to the image modality for object-presence detection.

### Results (Text Modality)

Final test set performance on ~10,000 unseen QA pairs:

| Metric | Score |
| --- | --- |
| Accuracy | 98.12% |
| Precision | 98.45% |
| Recall | 97.84% |
| F1-Score | 98.14% |

Comparison against baselines:

| Method | Accuracy | F1-Score |
| --- | --- | --- |
| POPE (original) | 88.1% | 86.5% |
| Pretrained TinyBERT (no fine-tuning) | 91.2% | 90.1% |
| TinyBERT + POPE (this work) | **98.12%** | **98.14%** |

The combination of TinyBERT's task capacity with POPE's clean binary supervision signal produced approximately a 10-point accuracy gain over the original POPE evaluation.

### Repository Structure
