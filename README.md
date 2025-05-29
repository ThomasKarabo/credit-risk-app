# 💳 Credit Risk Prediction Web App

This project is a Credit Risk Classification Web App built with a machine learning model that predicts whether a loan application is likely to be **Approved** or **Rejected** based on user input. The app is now fully built using **Streamlit** for fast and interactive deployment.

---

## 📊 Project Summary

- **Goal**: Predict loan approval status using applicant details.
- **Model**: Gradient Boosting Classifier (Train/Test Accuracy > 80%)
- **Frontend/Backend**: Python with Streamlit
- **Deployment**: Local or any platform that supports Streamlit apps (e.g., Streamlit Cloud, Render)

---

## 🧪 Model Development Process

### 1. Data Cleaning & Wrangling
- Removed outliers by analyzing boxplot distributions.
- Handled missing values and corrected inconsistencies.

### 2. Feature Engineering
- Filled missing values using `SimpleImputer`
- Scaled numerical features using `StandardScaler`
- Encoded categorical features using `OrdinalEncoder`

### 3. Handling Imbalanced Data
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset.
- Improved classification of rejected loan applications.

### 4. Model Training
- Built a pipeline using `sklearn.pipeline.Pipeline` for clean, reproducible training.
- Tested multiple classifiers and selected **Gradient Boosting Classifier**.

### 5. Model Serialization
- Trained model saved using `pickle` for fast loading and use in the Streamlit app.

---

## 🚀 Running the App Locally

### ✅ Prerequisites:
- Python 3.8+
- All libraries listed in `requirements.txt`

### 📦 Steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-risk-model.git```
2. Navigate into the project:
```
cd credit-risk-model
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the Streamlit app:
```
streamlit run streamlit_app.py
```
## 🙋🏽 About the Author
Thomas Karabo
Aspiring AI Engineer passionate about solving real-world problems using Data Science, Machine Learning, and streamlined automation tools like Streamlit.
