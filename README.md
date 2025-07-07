# 🧠 Breast Cancer Prediction using Neural Networks

This project builds a binary classification model using a neural network in **TensorFlow** to predict whether a tumor is **malignant** or **benign** based on features from the **Breast Cancer Wisconsin Diagnostic Dataset**.

---

## 📁 Dataset

- The dataset is loaded from `sklearn.datasets.load_breast_cancer()`.
- It contains **569** instances with **30 numeric features** and **1 target label**:
  - `0`: Malignant
  - `1`: Benign

---

## 🛠️ Tools & Libraries Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📊 Data Preprocessing

- Loaded the dataset and converted it into a Pandas DataFrame.
- Added a target column (`label`).
- Checked for null values (none found).
- Performed feature scaling using `StandardScaler`.

---

## 🧠 Model Architecture

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
```

- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Epochs: `50`
- Validation Split: `10%`

---

## 📈 Model Training & Evaluation

- Model achieved high accuracy with low loss over training epochs.
- Plotted accuracy and loss curves for both training and validation data.

**✅ Final Validation Accuracy: ~97.8%**

---

## 📦 Output Sample (Test Predictions)

Example prediction output:

```python
[1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 ...]
```

---

## 📉 Performance Visualization

### Accuracy Curve

> *(Insert accuracy plot image here if available)*

### Loss Curve

> *(Insert loss plot image here if available)*

---

## 📌 How to Run

```bash
# Clone this repo
git clone https://github.com/your-username/breast-cancer-prediction.git

# Install dependencies
pip install tensorflow scikit-learn pandas matplotlib

# Run the notebook in Google Colab or Jupyter
```

---

## 🙋‍♂️ Author

**Bharani Dharan**  
AI & Data Science Student | Passionate about ML & Deep Learning
