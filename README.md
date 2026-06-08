 Predict which telecom customers are likely to leave — end-to-end ML project with EDA, modeling, and Streamlit deployment.

---

## 🗂️ Project Structure

```
churn_project/
│
├── 1_eda.py            # Exploratory Data Analysis & visualizations
├── 2_preprocessing.py  # Data cleaning, encoding, scaling
├── 3_train_model.py    # Model training, SMOTE, evaluation
├── 4_app.py            # Streamlit web app for prediction
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **Source**: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043 customers
- **Target**: `Churn` (Yes/No)
- **Features**: 20 (demographics, services, billing)

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download from Kaggle and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder.

### 3. Run EDA
```bash
python 1_eda.py
```

### 4. Train the model
```bash
python 3_train_model.py
```
This generates `churn_model.pkl`, `scaler.pkl`, `feature_names.pkl`

### 5. Launch the web app
```bash
streamlit run 4_app.py
```

---

## 🧠 ML Pipeline

| Step | Details |
|------|---------|
| EDA | Churn rate, distributions, correlations |
| Preprocessing | Encoding, null handling, StandardScaler |
| Imbalance fix | SMOTE oversampling on training set |
| Models tried | Logistic Regression, Random Forest, Gradient Boosting |
| Evaluation | ROC-AUC, F1, Recall, Confusion Matrix |
| Deployment | Streamlit web app |

---

## 📈 Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.86 |
| **Gradient Boosting** | **~0.88** ✅ |

---

## 💡 Key Insights

- Month-to-month contracts have 3x higher churn rate than two-year contracts
- Customers with tenure < 12 months are at highest risk
- Fiber optic internet users churn more than DSL users
- Electronic check payment correlates with higher churn

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` · `matplotlib` · `seaborn` · `Streamlit`

---

*Built as a fresher Data Science portfolio project.*
