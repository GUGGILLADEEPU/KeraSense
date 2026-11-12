# KeraSense: Multimodal Deep Learning for Keratoconus Detection

**A medical image processing research project investigating multiview corneal analysis, feature fusion, and cross-dataset generalization for keratoconus classification.**

## 1. Project Overview

KeraSense is an academic deep learning project focused on classifying corneal imaging data into two categories: **Keratoconus (KC)** and **Non-Keratoconus (Non-KC)**.

Keratoconus is a progressive corneal disorder associated with changes in corneal shape and thickness. Corneal imaging systems provide multiple map representations of the eye, each describing different structural characteristics.

The project began with a seven-map deep learning classification pipeline using EfficientNet-B0. It was subsequently extended into a cross-dataset research study using a second imaging dataset, CornOrb, to investigate whether models trained on one corneal imaging dataset generalize to another.

The final research stage compares individual corneal maps with three-map feature fusion, evaluates within-dataset performance, investigates zero-shot cross-dataset transfer, and examines robustness when map features are unavailable.

**Research objective:** To investigate the contribution of multiple corneal map representations to keratoconus classification and evaluate how model performance changes across different imaging datasets.

---

## 2. Project Development: Two Experimental Stages

KeraSense developed through two major experimental stages. The architectures and reported results below belong to their respective stages and should not be interpreted as one continuous model evaluation.

### Stage 1 — Seven-Map EfficientNet-B0 Prototype

The initial project implemented a multimodal classification pipeline using seven complementary corneal maps.

**Architecture:**
- Seven corneal maps were supplied for each eye.
- A shared ImageNet-pretrained EfficientNet-B0 backbone extracted features from each map.
- Each map produced a 1,280-dimensional feature representation.
- The seven feature vectors were concatenated into an 8,960-dimensional representation.
- An MLP classifier predicted Keratoconus or Non-Keratoconus.
- Map-wise Grad-CAM visualizations were used to inspect regions influencing model predictions.

**Seven input maps:**

| Map identifier | Representation |
|---|---|
| `Sag_A` | Anterior sagittal curvature |
| `Sag_P` | Posterior sagittal curvature |
| `Elv_A` | Anterior elevation |
| `Elv_P` | Posterior elevation |
| `CT_A` | Corneal thickness representation |
| `EC_A` | Anterior eccentricity |
| `EC_P` | Posterior eccentricity |

**Initial prototype test results:**

| Metric | Result |
|---|---:|
| Accuracy | 93.33% |
| Sensitivity | 82.00% |
| Specificity | 99.00% |
| Precision (PPV) | 97.62% |
| Negative Predictive Value (NPV) | 91.67% |
| F1-score | 88.46% |

The prototype correctly classified 140 of 150 eyes, including 41 of 50 KC cases and 99 of 100 Non-KC cases.

These results belong to the initial seven-map EfficientNet-B0 experiment. Its held-out test set came from the same dataset source and does not represent external clinical validation.

### Stage 2 — Cross-Dataset Evaluation with CornOrb

The next research stage extended KeraSense beyond its original dataset.

The motivation was to investigate a practical limitation of medical imaging AI: a model that performs well on one imaging dataset may not retain the same performance when applied to images acquired using another system.

We introduced **CornOrb**, an Orbscan-based corneal imaging dataset, to investigate cross-dataset generalization.

The consolidated five-block experiment used a revised three-map pipeline based on frozen ResNet18 feature extraction and late feature fusion.

This stage is the primary experiment for the final cross-dataset research results documented below.

---

## 3. Datasets

### 3.1 KeraSense Dataset

Source: [Keratoconus Detection Dataset — Kaggle](https://www.kaggle.com/datasets/elmehdi12/keratoconus-detection)

The downloaded dataset contains:

- 4,011 images.
- 573 eyes.
- Seven corneal maps per eye.
- Original categories: Normal, Suspect and Keratoconus.

For the final binary experiment, Suspect cases were excluded.

| Partition | Eyes | Normal | Keratoconus |
|---|---:|---:|---:|
| Training | 240 | 120 | 120 |
| Validation | 60 | 30 | 30 |
| Independent test | 100 | 50 | 50 |
| **Total** | **400** | **200** | **200** |

The final three-map experiment therefore used 400 binary-classification eyes.

The KeraSense identifiers available for this experiment did not establish verified patient-level separation. The partitions should consequently be described as eye-level splits, not confirmed patient-independent splits.

### 3.2 CornOrb Dataset

CornOrb provides Orbscan-based corneal maps and associated clinical annotations.

The processed dataset used in the final experiment contained:

- 1,407 eyes.
- 5,628 map images across four map types.
- 886 Normal eyes.
- 521 Keratoconus eyes.

The final experiment used three map types corresponding to the selected KeraSense representations.

| Partition | Eyes | Normal | Keratoconus |
|---|---:|---:|---:|
| Training | 902 | 567 | 335 |
| Validation | 226 | 142 | 84 |
| Independent test | 279 | 177 | 102 |
| **Total** | **1,407** | **886** | **521** |

The original CornOrb clinical data contains 1,454 examination records. The 1,407 eyes reported here represent the matched and processed image cases used in our experiment.

The original CornOrb partition information was used to preserve patient separation between training and testing. The final paper should describe the exact matching and partition procedure and distinguish patient-level separation from eye-level counts.

### 3.3 Common Map Selection

To compare the datasets using conceptually corresponding inputs, three map types were selected.

| KeraSense map | CornOrb map | Representation |
|---|---|---|
| `Elv_A` | Anterior | Anterior elevation |
| `Elv_P` | Posterior | Posterior elevation |
| `CT_A` | Pachymetry | Corneal thickness |

The remaining four KeraSense maps were part of the original seven-map prototype but were not inputs to the final three-map cross-dataset experiment.

---

## 4. Final Model Architecture

### 4.1 Preprocessing

The selected maps were processed using a consistent image pipeline:

- Image loading and RGB conversion.
- Resizing to 224 × 224 pixels.
- Tensor conversion.
- ImageNet normalization during model input preparation.

The three maps were retained as separate inputs to preserve their individual representations.

### 4.2 ResNet18 Feature Extraction

The final experiment used an ImageNet-pretrained ResNet18 architecture.

The pretrained feature-extraction backbone was frozen, and trainable classification heads were used for the classification experiments.

Each map generated a 512-dimensional feature vector:

`Anterior features: 512 dimensions`

`Posterior features: 512 dimensions`

`Pachymetry features: 512 dimensions`

### 4.3 Single-Map Classification

Separate classifiers were trained using each map:

1. Anterior elevation.
2. Posterior elevation.
3. Pachymetry.

This allowed the predictive contribution of individual map representations to be evaluated.

### 4.4 Three-Map Feature Fusion

The three 512-dimensional feature vectors were concatenated:

`512 + 512 + 512 = 1,536 dimensions`

The resulting 1,536-dimensional representation was passed to a trainable fusion classifier.

This approach is called **late feature fusion**, because each map is encoded separately before the representations are combined.

### 4.5 Classification

The classification heads produced a binary prediction:

- Normal.
- Keratoconus.

The final research experiment used a fixed decision threshold of 0.50 for threshold-based test metrics.

---

## 5. Training Configuration

The consolidated five-block experiment used the following settings:

| Parameter | Configuration |
|---|---|
| Feature extractor | ImageNet-pretrained ResNet18 |
| Backbone | Frozen |
| Feature dimensions per map | 512 |
| Fusion representation | 1,536 dimensions |
| Classifier | Trainable multilayer classification head |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Batch size | 64 |
| Maximum epochs | 40 |
| Early stopping patience | 8 epochs |
| Loss function | Binary cross-entropy with logits |
| Model selection | Best validation AUC |
| Random seed | 42 |
| Final evaluation threshold | 0.50 |

The final five-block experiment was executed using CPU-based PyTorch in the available runtime. The earlier prototype's GPU training configuration should not be attributed to this later run.

---

## 6. Experimental Design

The final research stage included three evaluation settings.

### Experiment 1 — KeraSense Independent Test

Models were trained using the KeraSense training partition and evaluated on its held-out independent test partition.

**Purpose:** To assess performance on unseen eyes from the same dataset source.

### Experiment 2 — CornOrb Native Independent Test

Models were trained using the CornOrb training partition and evaluated on its held-out independent test partition.

**Purpose:** To assess performance within the second imaging dataset.

### Experiment 3 — KeraSense-to-CornOrb Zero-Shot Transfer

Models trained on KeraSense were directly evaluated on CornOrb without retraining or fine-tuning on CornOrb labels.

**Purpose:** To investigate cross-dataset generalization under a change in imaging dataset and acquisition domain.

### Experiment 4 — Missing-Map Feature Neutralization

The three-map fusion models were evaluated under seven input configurations:

- Anterior map neutralized.
- Posterior map neutralized.
- Pachymetry map neutralized.
- Anterior and posterior neutralized.
- Anterior and pachymetry neutralized.
- Posterior and pachymetry neutralized.
- All three maps neutralized.

For this experiment, the selected feature vectors were replaced with the corresponding training-split mean feature vector.

This is a controlled feature-neutralization experiment. It is not equivalent to naturally missing clinical images.

---

## 7. Final Experimental Results

The following results belong to the consolidated five-block ResNet18 experiment. They are separate from the initial seven-map EfficientNet-B0 prototype results.

### 7.1 KeraSense Independent Test Results

Test size: 100 eyes.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1-score | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Anterior | 50.00% | 100.00% | 0.00% | 50.00% | 0.6667 | 0.9292 |
| Posterior | 88.00% | 76.00% | 100.00% | 100.00% | 0.8636 | 0.9880 |
| Pachymetry | 88.00% | 78.00% | 98.00% | 97.50% | 0.8667 | 0.9556 |
| **Three-map fusion** | **96.00%** | **98.00%** | **94.00%** | **94.23%** | **0.9608** | **0.9940** |

The three-map fusion model correctly classified 96 of 100 eyes at the fixed 0.50 threshold.

It detected 49 of 50 KC cases and correctly classified 47 of 50 Normal cases.

### 7.2 CornOrb Native Independent Test Results

Test size: 279 eyes.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1-score | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Anterior | 93.55% | 82.35% | 100.00% | 100.00% | 0.9032 | 0.9970 |
| Posterior | 94.27% | 84.31% | 100.00% | 100.00% | 0.9149 | 0.9795 |
| Pachymetry | 93.55% | 84.31% | 98.87% | 97.67% | 0.9053 | 0.9596 |
| **Three-map fusion** | **97.13%** | **99.02%** | **96.05%** | **93.40%** | **0.9619** | **0.9951** |

The CornOrb-native fusion model correctly classified 271 of 279 eyes.

These results represent performance within the CornOrb dataset and should not be interpreted as external clinical validation.

### 7.3 KeraSense-to-CornOrb Zero-Shot Results

The KeraSense-trained models were evaluated directly on the CornOrb independent test set.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1-score | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Anterior | 36.56% | 100.00% | 0.00% | 36.56% | 0.5354 | 0.8495 |
| Posterior | 86.02% | 69.61% | 95.48% | 90.00% | 0.7845 | 0.8814 |
| Pachymetry | 53.76% | 32.35% | 66.10% | 35.48% | 0.3385 | 0.4827 |
| **Three-map fusion** | **42.29%** | **99.02%** | **9.60%** | **38.77%** | **0.5565** | **0.9298** |

The zero-shot fusion model achieved an AUC of 0.9298 but only 42.29% accuracy at the fixed 0.50 threshold.

Its confusion matrix was:

| Actual / Predicted | Normal | Keratoconus |
|---|---:|---:|
| Normal | 17 | 160 |
| Keratoconus | 1 | 101 |

This demonstrates a substantial difference between ranking performance and fixed-threshold classification performance.

The model retained useful score-ranking information, as reflected by its AUC, but produced a large number of false-positive predictions at the 0.50 threshold.

The result indicates substantial cross-dataset score-distribution and threshold-level shift. It does not establish that the model is clinically transferable without adaptation or calibration.

### 7.4 Missing-Map Feature-Neutralization Results

Selected KeraSense-to-CornOrb results:

| Neutralized maps | Accuracy | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| Anterior | 36.56% | 100.00% | 0.00% | 0.8663 |
| Posterior | 49.46% | 97.06% | 22.03% | 0.8755 |
| Pachymetry | 42.65% | 100.00% | 9.60% | 0.9614 |
| Anterior + posterior | 36.56% | 100.00% | 0.00% | 0.4755 |
| Anterior + pachymetry | 36.56% | 100.00% | 0.00% | 0.9333 |
| Posterior + pachymetry | 53.41% | 100.00% | 26.55% | 0.9455 |
| All three | 36.56% | 100.00% | 0.00% | 0.5000 |

The results show that model behavior changes according to which map features are neutralized.

When all three feature vectors are neutralized, the input representation becomes constant and AUC is 0.50, as expected for tied predictions.

These experiments are useful for understanding feature dependence under controlled neutralization, but do not establish clinical map importance or natural missing-input robustness.

---

## 8. Main Research Findings

The final research stage produced several findings:

1. **Three-map fusion performed strongly within both datasets.** The KeraSense fusion model achieved 96.00% accuracy, while the CornOrb-native fusion model achieved 97.13% accuracy.

2. **Individual maps behaved differently.** Posterior elevation and pachymetry produced different classification and ranking performance across the two datasets.

3. **Cross-dataset transfer showed a significant threshold-level performance drop.** The KeraSense-trained fusion model achieved 0.9298 AUC on CornOrb but only 42.29% accuracy at the fixed 0.50 threshold.

4. **Map neutralization changed prediction behavior.** The missing-map experiments demonstrated that model outputs depend on the available feature representations.

5. **External evaluation matters.** Strong within-dataset performance does not automatically establish generalization across imaging systems.

The central research finding is the contrast between strong within-dataset results and substantial threshold-level degradation during zero-shot cross-dataset transfer.

---

## 9. Explainability and Model Inspection

The initial seven-map EfficientNet-B0 prototype included map-wise Grad-CAM visualizations.

Grad-CAM was used to inspect image regions associated with model predictions. These visualizations can support qualitative model inspection and error analysis.

The Grad-CAM functionality belongs to the initial prototype. It should not be described as an evaluated component of the final three-map ResNet18 experiment unless the corresponding implementation and outputs are included in that experiment's archived artifacts.

Grad-CAM maps are not clinical segmentation masks and do not establish causal explanations for predictions.

---

## 10. Research Motivation and Scope

The project addresses a practical problem in medical image processing: deep learning models can perform well on the dataset used for development but may behave differently when applied to another imaging dataset.

Corneal imaging systems can differ in acquisition characteristics, map rendering, measurement distributions and clinical data composition. These differences can affect model generalization.

KeraSense investigates this issue by comparing two corneal imaging datasets and evaluating both native performance and direct zero-shot transfer.

The project provides a basis for further research into:

- Cross-device and cross-dataset generalization.
- Domain adaptation for corneal imaging.
- Validation-based probability calibration.
- Patient-level evaluation.
- Map-wise feature contribution.
- Robustness to missing imaging inputs.
- Evaluation on additional independent clinical datasets.

The project does not claim that multiview fusion or keratoconus deep learning classification is new by itself. Its research emphasis is the combined evaluation of map-specific models, three-map fusion, cross-dataset transfer and missing-feature behavior.

---

## 11. Limitations

The following limitations should be considered when interpreting the results:

- **KeraSense patient separation:** Available identifiers do not establish verified patient-level separation across all partitions. The final KeraSense split is reported at eye level.
- **CornOrb reference labels:** CornOrb labels incorporate clinical records together with Orbscan findings. They are not an independent clinical gold standard fully separate from the imaging information.
- **Confounded domain shift:** The two datasets differ in more than imaging hardware. Site, population, acquisition and annotation differences may also contribute.
- **Test-set size:** The KeraSense independent test contains 100 eyes, which limits the precision of performance estimates.
- **Fixed decision threshold:** The zero-shot evaluation used a fixed 0.50 threshold. It was not optimized on the independent test set.
- **Feature neutralization:** Mean-feature replacement is a controlled experiment and does not represent naturally missing corneal maps.
- **Binary classification:** Suspect cases were excluded from the final binary experiment. The results do not establish performance on early, borderline or forme-fruste keratoconus.
- **Frozen feature extraction:** The final ResNet18 experiment uses frozen pretrained feature extractors with trainable classification heads, rather than full end-to-end fine-tuning.
- **Clinical applicability:** The models have not undergone prospective clinical validation.

---

## 12. Future Work

Potential next research directions include:

- Patient-level validation using verified patient identifiers.
- Confidence intervals and patient-clustered uncertainty estimates.
- Calibration fitted using validation data only.
- Domain adaptation and feature alignment across imaging systems.
- Evaluation across more imaging devices and independent clinical cohorts.
- Natural missing-input training and testing.
- Additional map-fusion and attention-based architectures.
- Evaluation of Normal, Suspect and Keratoconus as three separate classes.
- Further error analysis and explainability evaluation.

---

## 13. Technologies Used

### Initial seven-map prototype
- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- OpenCV
- Pillow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Grad-CAM
- Jupyter Notebook
- IPyWidgets

### Final cross-dataset research stage
- Python
- PyTorch
- Torchvision
- ResNet18
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Google Colab

---

## 14. Repository Structure

The following is the intended project organization. The actual repository should be checked before claiming that every listed directory or application file is already committed.

```text
KeraSense/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── miprerunresearchfinal.ipynb
├── models/
│   ├── initial_efficientnet/
│   └── final_resnet18/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── results/
│   ├── initial_prototype/
│   └── final_cross_dataset/
├── assets/
│   ├── architecture/
│   ├── corneal_maps/
│   └── figures/
└── docs/
    ├── methodology.md
    └── limitations.md
```

The final five-block research archive contains the consolidated model checkpoints, feature caches, evaluation metrics, predictions, figures, methods and limitations notes, and final audit files.

The archive was created in the Colab runtime at:

`/content/KERASENSE_RESEARCH/KERASENSE_FINAL_RESEARCH_ARCHIVE.zip`

The original raw datasets are not duplicated inside that archive.

---

## 15. Reproducibility

The final consolidated research artifacts include:

- Eight trained model checkpoints.
- KeraSense and CornOrb feature caches.
- Independent-test and zero-shot predictions.
- Main evaluation metrics.
- Missing-map evaluation results.
- Validation results.
- Research figures.
- Methods and limitations documentation.
- Final audit information.

The final archive should be retained alongside the notebook and the dataset source references.

For reproducibility, the exact data partitions, preprocessing configuration, model checkpoints and corresponding result files must be used together. Results from the initial EfficientNet-B0 prototype must remain separate from the final ResNet18 cross-dataset experiment.

---

## 16. Disclaimer

KeraSense is an academic research project intended for educational and experimental medical image processing research.

It is not a medical device and has not been clinically validated. Its predictions must not be used as a substitute for professional ophthalmic examination, diagnosis or treatment.

## Author

**Deepu Guggilla**

