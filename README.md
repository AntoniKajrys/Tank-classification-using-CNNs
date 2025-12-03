
# 🛡️ Tank Classification from RGB Images

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2B-orange.svg)](https://pytorch.org/)
[![Report](https://img.shields.io/badge/Report-PDF-green.svg)](<Project%20Report.pdf>)

This project develops a deep learning model to classify military tanks by country of origin (**American**, **Russian**, **Israeli**) from RGB images. It addresses friendly fire risks in unmanned systems by enabling reliable target identification.

The primary model is a **fine-tuned ResNet50 CNN** achieving **94.16% test accuracy**. A hybrid **CNN + Random Forest** baseline reaches **86.7%**.

## 🎯 Problem Statement

Classify tanks into 3 classes using supervised learning on 3244 filtered RGB images (256x256 resolution, 65k features per image). Dataset imbalance: Israeli (1752), Russian (946), American (546).

**Dataset Sources** (Appendix in [Report](Project_Report.pdf)):

- Roboflow & Kaggle datasets (links in report).
- Merged, filtered (remove non-target tanks), relabeled, resized.

**Splits** (stratified):

| Split           | American      | Russian        | Israeli        | Total          |
| --------------- | ------------- | -------------- | -------------- | -------------- |
| Train           | 445           | 803            | 1217           | 2465           |
| Val             | 53            | 123            | 320            | 496            |
| Test            | 48            | 114            | 215            | 377            |
| **Total** | **546** | **1040** | **1752** | **3338** |

Data: [`data_combined/train|val|test/`] – Images named `{class}_{index}.jpg`.

## 📊 Results Summary

### CNN (Fine-tuned ResNet50)

| Set  | Accuracy | American (P/R/F1) | Russian (P/R/F1) | Israeli (P/R/F1) |
| ---- | -------- | ----------------- | ---------------- | ---------------- |
| Val  | 96.98%   | 0.96/0.89/0.92    | 0.95/0.98/0.97   | 0.98/0.98/0.98   |
| Test | 94.16%   | 1.00/0.62/0.77    | 0.90/0.98/0.94   | 0.96/0.99/0.97   |

### Random Forest (on ResNet Features)

| Set  | Accuracy | American (P/R/F1) | Russian (P/R/F1) | Israeli (P/R/F1) |
| ---- | -------- | ----------------- | ---------------- | ---------------- |
| Test | 86.7%    | 0.72/0.38/0.49    | 0.80/0.99/0.88   | 0.93/0.91/0.92   |

CNN outperforms RF, especially on underrepresented American tanks. Overfitting noted; future: weighted CE loss, augmentation.

**Full Report**: [Project Report.pdf](Project_Report.pdf)

## 🚀 Quick Start

1. **Clone & Install**:
   ```bash
   git clone 
   cd tank-classification
   pip install -r requirements.txt
   ```
