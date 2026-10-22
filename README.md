# KeraSense: Multimodal AI for Keratoconus Detection

**An explainable deep learning framework for keratoconus classification using seven complementary corneal maps.**

KeraSense is an academic deep learning project that integrates multiple corneal imaging representations to classify eyes as **Keratoconus (KC)** or **Non-Keratoconus (Non-KC)**. It combines shared EfficientNet-B0 feature extraction, feature-level fusion, and map-wise Grad-CAM visualizations in a unified pipeline.

## Key Features

- **Multimodal corneal analysis:** Processes seven complementary corneal maps per eye.
- **Efficient feature extraction:** Uses a shared EfficientNet-B0 backbone.
- **Feature-level fusion:** Combines seven 1,280-dimensional feature vectors into an 8,960-dimensional representation.
- **Binary classification:** Predicts KC or Non-KC.
- **Explainable AI:** Uses map-wise Grad-CAM to inspect regions influencing predictions.
- **Quantitative evaluation:** Reports accuracy, sensitivity, specificity, precision, negative predictive value, and F1-score.

## Model Architecture

The KeraSense pipeline consists of six stages:

1. **Input:** Seven corneal maps for each eye.
2. **Preprocessing:** RGB conversion, resizing to 224 × 224 pixels, tensor conversion, and ImageNet normalization.
3. **Feature extraction:** A shared EfficientNet-B0 backbone extracts 1,280 features per map.
4. **Feature fusion:** The seven feature vectors are concatenated into an 8,960-dimensional representation.
5. **Classification:** A multilayer perceptron (MLP) predicts KC or Non-KC.
6. **Interpretability:** Map-wise Grad-CAM visualizations help inspect model predictions.

## Corneal Maps

| Map identifier | Representation |
|---|---|
| `Sag_A` | Anterior sagittal curvature |
| `Sag_P` | Posterior sagittal curvature |
| `Elv_A` | Anterior elevation |
| `Elv_P` | Posterior elevation |
| `CT_A` | Corneal thickness representation |
| `EC_A` | Anterior eccentricity |
| `EC_P` | Posterior eccentricity |

## Dataset

The project uses the [Keratoconus Detection dataset on Kaggle](https://www.kaggle.com/datasets/elmehdi12/keratoconus-detection).

The downloaded dataset contains **4,011 images representing 573 eyes**, with seven maps per eye.

For the current binary classification experiment:

- **Development set:** 423 eyes, divided into training and validation sets.
- **Independent test set:** 150 eyes.
- **Non-KC:** Normal and Suspect categories combined.
- **KC:** Keratoconus category.

The test set is held out from training and validation, but it originates from the same dataset source. It is not external clinical validation.

## Experimental Results

The current model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 93.33% |
| Sensitivity | 82.00% |
| Specificity | 99.00% |
| Precision (PPV) | 97.62% |
| Negative Predictive Value (NPV) | 91.67% |
| F1-score | 88.46% |

The model correctly classified **140 of 150 eyes**. It detected 41 of 50 KC eyes and correctly classified 99 of 100 Non-KC eyes.

Sensitivity remains an important area for improvement because some KC cases were missed. These findings represent experimental performance on the selected dataset and do not establish clinical readiness.

## Explainability

KeraSense uses map-wise Grad-CAM to visualize image regions influencing individual predictions. These visualizations support model inspection and error analysis; they are not clinical segmentation masks or proof of causal reasoning.

## Technologies

- Python
- PyTorch and Torchvision
- EfficientNet-B0
- OpenCV and Pillow
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Grad-CAM
- Jupyter Notebook and IPyWidgets

## Repository Structure

```text
KeraSense/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── models/
├── src/
│   ├── model.py
│   └── predict.py
├── notebooks/
├── assets/
└── docs/
```

The directory structure may evolve as the project is organized for reproducibility and deployment.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/GUGGILLADEEPU/KeraSense.git
cd KeraSense
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows, activate it using:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

Follow the instructions provided with the uploaded notebook or application. The model architecture, checkpoint, preprocessing, and input ordering must match the original training implementation.

## Limitations and Future Work

- Improve sensitivity and investigate false-negative cases.
- Compare alternative feature-fusion strategies.
- Conduct controlled map-wise ablation experiments.
- Evaluate robustness across multiple data splits and random seeds.
- Validate the model on independent external datasets.
- Further investigate explainability and potential clinical applicability.

## Disclaimer

KeraSense is an academic research project intended for educational and experimental purposes. It is **not a medical device** and must not be used as a substitute for professional ophthalmic examination, diagnosis, or treatment.

## Author

**Deepu Guggilla**

GitHub: [@GUGGILLADEEPU](https://github.com/GUGGILLADEEPU)
