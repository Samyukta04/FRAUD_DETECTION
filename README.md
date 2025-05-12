# Credit Card Fraud Detection using Neural Networks

This project implements a binary classification model to detect fraudulent credit card transactions using a fully connected neural network. The dataset is highly imbalanced, so techniques like SMOTE and precision-recall evaluation are used to improve performance.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Contains transactions made by European cardholders in September 2013.
- **Features**:
  - 30 columns: `V1` to `V28` (PCA-transformed), `Amount`, `Time`, and `Class`
  - `Class` = 1 for fraud, 0 for normal transactions

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Dropped `Time` column
   - Scaled `Amount` and all features using `StandardScaler`

2. **Train-Test Split**
   - Stratified split (80% training, 20% testing)
   - Ensured no data leakage by applying SMOTE only on training data

3. **Handling Class Imbalance**
   - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples in the training set

4. **Model Architecture**
   - A simple Dense Neural Network:
     - Dense → Dropout → Dense → Dropout → Dense (Sigmoid)
   - Optimizer: `Adam`
   - Loss: `Binary Crossentropy`

5. **Evaluation Metrics**
   - **Confusion Matrix**
   - **Classification Report**
   - **Precision-Recall AUC**
   - **PR Curve Plot**

---

## 📊 Results

- Achieved high **Recall** and **PR AUC** (especially important for fraud detection)
- Adjusted classification threshold from 0.5 to **0.2** to reduce false negatives

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

**requirements.txt**

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
tensorflow
```

---

## 🧠 Model Training

```bash
python fraud_detection_nn.py
```

---

## 📈 Output

* PR AUC Score
* Confusion Matrix (Heatmap)
* Precision-Recall Curve
* Classification Report

