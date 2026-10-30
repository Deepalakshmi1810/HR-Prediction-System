# HR Prediction System

## Project Overview
The HR Prediction System is a data analytics and machine learning project that analyzes candidate information and predicts whether a candidate is likely to be Placed or Not Placed.

## Objectives
- Analyze candidate placement data.
- Clean and preprocess the dataset.
- Store candidate records in MySQL.
- Build a machine learning classification model.
- Visualize insights through a Streamlit dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- SQL and MySQL
- Scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook

## Machine Learning Algorithm
**Logistic Regression** was used for binary classification.

### ML Workflow
1. Data cleaning and preprocessing
2. Handling missing values
3. Encoding categorical variables
4. Feature scaling
5. Train-test split (80:20)
6. Model training
7. Model evaluation
8. Prediction

## Model Performance
- Accuracy: 88.07%
- ROC-AUC Score: 0.9456

## Dataset Summary
- Total records after cleaning: 50,124
- Total features: 26
- Duplicate records after cleaning: 0
- Missing values after cleaning: 0

## Project Files
- `hr_prediction.py` – Data processing, MySQL integration and ML model.
- `app.py` – Streamlit dashboard.
- `requirements.txt` – Required Python libraries.

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

## Conclusion
This project demonstrates how Python, SQL, machine learning and data visualization can be combined to analyze candidate placement outcomes and generate useful insights.

---
**Project: HR Prediction System**