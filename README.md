# 🧠 Stroke Prediction Neural Network

This project implements a neural network using **PyTorch** to predict the likelihood of a stroke based on patient health data. The model is trained on the **"Healthcare Dataset Stroke Data"**, which includes various patient features like age, BMI, smoking status, and glucose level.


---

## 📊 Dataset

The dataset includes the following key features:

- **gender**
- **age**
- **hypertension**
- **heart_disease**
- **ever_married**
- **work_type**
- **Residence_type**
- **avg_glucose_level**
- **bmi**
- **smoking_status**
- **stroke** (target)

---

## 🔄 Data Preprocessing

- Missing values in `bmi` are filled with the **mean**.
- Categorical features are encoded using **Label Encoding**.
- Features are **scaled**:
  - `age` divided by 100
  - `avg_glucose_level` divided by 1000
  - `bmi` divided by 100
- `id` and `stroke` columns are removed from input features.
- `stroke` is kept as the target variable.

---

## 🧠 Model Architecture

A simple feed-forward neural network:

- Input Layer: 10 features
- Hidden Layer 1: 32 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output Layer: 1 neuron (binary classification)


---

## 📦 Requirements

- `torch`
- `torchmetrics`
- `pandas`
- `numpy`
- `scikit-learn`

Install with:

```bash
pip install torch torchmetrics pandas numpy scikit-learn
