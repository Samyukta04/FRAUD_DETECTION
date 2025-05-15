# 💳 Credit Card Fraud Detection using Neural Networks

This project implements multiple neural network models to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, we focus on careful preprocessing, use of SMOTE for oversampling, and evaluation using precision-recall-based metrics.

---

## 📁 Dataset

* **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Features:** PCA-transformed features (V1–V28), `Amount`, and `Class` (fraud = 1, normal = 0)
* **Size:** \~285,000 transactions; \~0.17% are fraud cases

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Dropped `Time` column
* Scaled `Amount` and all features using `StandardScaler`

### 🔹 Train-Test Split

* Stratified 80/20 split
* SMOTE applied **only to the training set** to avoid data leakage

---

## 🧠 Models Implemented

### 1. **Multilayer Perceptron (MLP)**

* Implemented using `sklearn.neural_network.MLPClassifier`
* Tuned hidden layers: (50, 30) and (30, 50, 20)
* Best performance achieved with 3 hidden layers
* High precision and recall after threshold tuning

### 2. **Feedforward Neural Network (FNN)**

* Implemented from scratch using NumPy
* Two hidden layers with ReLU activation
* Used manual backpropagation and batch gradient descent
* Tuned using grid search for learning rate and architecture

### 3. **LSTM (Long Short-Term Memory)**

* Sequential model treating each transaction as a time step
* One LSTM layer followed by a dense output layer
* Designed to capture temporal dependencies between patterns
* Tuned with different sequence lengths and hidden states

### 4. **Autoencoder with LightGBM**

* Unsupervised Autoencoder trained to reconstruct normal transactions
* Reconstruction error used as a new feature
* LightGBM classifier trained on these features for final classification
* Achieved best PR AUC among all models

---

## 📊 Evaluation Metrics

* **Precision, Recall, F1-Score**
* **PR AUC (Precision-Recall Area Under Curve)**
* **Confusion Matrix**
* **Precision-Recall Curve Plot**

**Threshold tuned from 0.5 → 0.2** to reduce false negatives

---

## ✅ Results

| Model              | PR AUC | Remarks                                |
| ------------------ | ------ | -------------------------------------- |
| Autoencoder + LGBM | High   | Best overall performance               |
| LSTM               | High   | Good sequence-based fraud detection    |
| MLP                | High   | Tuned well, interpretable architecture |
| FNN                | Medium | Solid baseline, handcrafted backprop   |

---

## 🚀 Run the Model

```bash
python fraud_detection_nn.py
```

---

## 📈 Output

* PR AUC Score
* Classification Report
* Confusion Matrix (heatmap)
* Precision-Recall Curve

---

