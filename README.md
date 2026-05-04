# A Research Study on Multimodal Hallucinations

A supervised hallucination detection system for text and image modalities, built by adapting the POPE (Polling-based Object Probing Evaluation) framework. The text pipeline (TinyBERT) achieved 98.12% accuracy, outperforming the POPE baseline by approximately 10 percentage points. The image pipeline (CLIP-ViT-B/16) achieved 92.4% validation accuracy, outperforming the POPE baseline by approximately 4 points.

This work was completed as a group research project for the MSc Data Science and AI programme at the University of Liverpool (Semester 1, 2025).

> **Repository scope.** This repository contains the project write-up, results, and the image-pipeline training script (`model_training.py`). Other components of the pipeline — text preprocessing, POPE-style question generation, the TinyBERT training code, and SEEM-based image annotation — are not included in this public release.

## Overview

Multimodal Large Language Models (MLLMs) frequently produce hallucinations — fluent but factually inaccurate content inconsistent with the input. While the POPE framework introduced a useful binary probing approach for evaluating object hallucination, it remains a discriminative evaluation tool rather than a trainable detection system.

This project adapts POPE into a supervised detection pipeline that operates across text and image modalities, with the goal of producing a simple, reliable, and resource-efficient detector.

## Approach

The system reformulates hallucination detection as a binary classification task. For a given input (text document or image) and a candidate entity, the model predicts whether that entity is genuinely present or hallucinated.

### Text Pipeline

- **Source:** ~11,100 long-form documents, filtered to ~10,000 usable documents
- **Entity extraction:** capitalized words and frequent nouns, with strict token-boundary matching
- **POPE-style QA generation:** ~100,000 binary (entity-present / entity-absent) question pairs
- **Model:** pretrained TinyBERT fine-tuned with a linear classification head on the [CLS] token
- **Input format:** `[CLS] question [SEP] document [SEP]`, 256-token cap with leading-paragraph truncation
- **Training:** 5 epochs on consumer hardware (MacBook CPU), class-balanced sampler to address label imbalance

### Image Pipeline

- **Source:** 84,000 images, split 80/20 into training and validation sets
- **Annotation:** automated object detection using SEEM (Segment Everything Everywhere All at Once), removing the manual annotation bottleneck of the original POPE protocol
- **POPE-style question generation:** six binary questions per image (three positive, three negative), drawing negatives from random, popular, and adversarial categories — yielding ~504,000 question pairs across 80+ COCO object classes
- **Model:** CLIP-ViT-B/16 image and text encoders with element-wise feature fusion, followed by a linear classification head
- **Loss / optimiser:** binary cross-entropy with logits, AdamW (learning rate 5e-5)
- **Training:** 5 epochs on consumer hardware (MacBook Air M1/M2), batch size 4, with gradient clipping for training stability

## Results

### Text Modality

Final test set performance on ~10,000 unseen QA pairs:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.12% |
| Precision | 98.45% |
| Recall    | 97.84% |
| F1-Score  | 98.14% |

Comparison against baselines:

| Method                              | Accuracy | F1-Score |
|-------------------------------------|----------|----------|
| POPE (original)                     | 88.1%    | 86.5%    |
| Pretrained TinyBERT (no fine-tuning)| 91.2%    | 90.1%    |
| TinyBERT + POPE (this work)         | 98.12%   | 98.14%   |

The combination of TinyBERT's task capacity with POPE's clean binary supervision signal produced approximately a 10-point accuracy gain over the original POPE evaluation.

### Image Modality

Best validation performance (epoch 3 of 5):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 92.4%  |
| Precision | 93.1%  |
| Recall    | 89.6%  |
| F1-Score  | 91.2%  |

Comparison against the POPE baseline:

| Method                       | Accuracy | F1-Score |
|------------------------------|----------|----------|
| POPE (original)              | 88.1%    | 86.5%    |
| CLIP + POPE (this work)      | 92.4%    | 91.2%    |

Performance varied by object category, with frequent COCO classes (e.g., person, car) reaching ~94% accuracy and adversarial negatives — semantically related but absent objects — reaching ~90%.

> Note: text-modality figures are reported on a held-out test set; image-modality figures are best validation results. A held-out test evaluation for the image pipeline is planned as part of the code release.

## Repository Contents

- `model_training.py` — image-pipeline training script (CLIP-ViT-B/16 + binary classifier). Implementation authored by Krishna Kethineni Ramesh.
- `README.md` — project write-up and results

The script expects a POPE-style annotation JSON file (`--train_annotations`, `--val_annotations`) and an image folder (`--image_folder`). The annotation generation code and dataset are not included in this public release.

## Team and Contributions

This was a group project. Team members:

- Avanish Deshpande
- Hari Prasath Somasundaram
- Harsha Vardhan Arangi
- Krishna Kethineni Ramesh
- Preksha Khera

**My contribution (Harsha Vardhan Arangi):** designed and implemented the text and image detection pipelines — preprocessing, POPE-style QA generation, TinyBERT fine-tuning, and evaluation — and co-authored the project report.Led the dataset collection effort, contributed to the project's conceptual framing (overview of multimodal LLMs, problem statement on hallucination, motivation for detection), assisted teammates with code where needed, and co-authored the project report.

## References

1. Li et al. (2023). *Evaluating Object Hallucination in Large Vision-Language Models* (POPE). EMNLP 2023.
2. Jiao et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.* Findings of EMNLP 2020.
3. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision* (CLIP). ICML 2021.
4. Zou et al. (2023). *Segment Everything Everywhere All at Once* (SEEM). NeurIPS 2023.
