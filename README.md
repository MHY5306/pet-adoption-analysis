# 🐾 Pet Adoption Prediction (with Image Features)

## 📌 Project Overview

This project aims to predict pet adoption speed using both structured data and image features.

We compare:

* A baseline model using tabular features
* A multimodal approach combining tabular and image features

👉 Goal: Evaluate whether image information improves prediction performance.

---

## 📊 Dataset

* Source: Kaggle PetFinder Adoption Prediction
* Includes:

  * Pet metadata (Age, Fee, Health, Type, etc.)
  * Pet images

Target Variable:

fast_adopt = AdoptionSpeed <= 1

Due to size and licensing restrictions, the dataset is not included in this repository.
You can download it from:
https://www.kaggle.com/c/petfinder-adoption-prediction/data

---

## 🧪 Workflow

### 1. Data Loading

* Load dataset (train.csv)
* Inspect data structure and missing values

---

### 2. Exploratory Data Analysis (EDA)

Key observations:

* Younger pets are adopted faster
* Adoption fee has limited impact
* Health condition influences adoption probability

---

### 3. Feature Engineering

df["fast_adopt"] = df["AdoptionSpeed"] <= 1
df["is_young"] = df["Age"] <= 6
df["is_free"] = df["Fee"] == 0
df["is_healthy"] = df["Health"] == 1
df["Type"] = df["Type"] - 1  # Dog=0, Cat=1

Final structured features:

* is_young
* is_free
* is_healthy
* Type
* PhotoAmt

---

### 4. Baseline Model

Model:

* Logistic Regression

model = LogisticRegression(max_iter=1000)

Performance:

* Accuracy: ~0.76

---

### 5. Image Feature Extraction

We used a pretrained CNN:

* Model: ResNet18 (PyTorch)
* Output: 512-dimensional feature vector

Process:

* Resize image to 224x224
* Convert to tensor
* Extract features without gradient

feature = model(img_tensor)
feature = feature.view(-1).numpy()

---

### 6. Multimodal Model

We combined:

* Structured features
* Image features (512 dimensions)

Then trained Logistic Regression again.

---

### 7. Model Evaluation

Evaluation methods:

* Train-test split
* 5-fold Cross Validation

cross_val_score(model, X, y, cv=5)

---

## 📈 Results

| Model                   | Accuracy |
| ----------------------- | -------- |
| Baseline                | ~0.76    |
| With Image Features     | ~0.76    |
| Cross Validation (mean) | ~0.766   |

---

## 🔍 Key Insights

* Structured features (age, health, type) are the primary drivers of adoption
* Image features extracted from pretrained CNNs did not significantly improve performance
* Adding more features does not guarantee better results
* Feature relevance is more important than feature quantity

---

## 🧠 Discussion

This project highlights an important concept in machine learning:

👉 More data ≠ better performance

Possible reasons why image features did not help:

* CNN features are generic and not aligned with adoption behavior
* Adoption decisions depend more on metadata than visual appearance
* Lack of fine-tuning on domain-specific data

---

## 🛠 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* PyTorch / Torchvision
* Matplotlib

---

## 📁 Project Structure

pet-adoption-analysis/
│
├── 01_data_loading.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_model_baseline.ipynb
├── 05_image_analysis.ipynb
├── 06_image_insight.ipynb
└── README.md

---

## 🚀 Future Improvements

* Apply PCA to reduce image feature dimensions
* Fine-tune CNN instead of using pretrained features directly
* Try tree-based models (XGBoost, LightGBM)
* Combine structured + image features using deep learning
* Add explainability methods (SHAP)

---

## 💡 Key Takeaway

👉 Feature relevance matters more than feature quantity
👉 Validation is essential for reliable conclusions
👉 Not all additional data sources improve model performance
