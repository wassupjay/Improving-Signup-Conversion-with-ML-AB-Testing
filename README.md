# Improving Signup Conversion Using ML and A/B Testing

This project simulates a real-world marketing experiment where machine learning is used to personalize call-to-action (CTA) messages and A/B testing is conducted to validate the effectiveness of the personalization.

## 📊 Dataset
A synthetic dataset of 10,000 users with the following columns:
- `device_type`, `browser`, `traffic_source`
- `scroll_depth`, `time_on_page`
- `signed_up` (target variable)

## 🧠 Machine Learning
- Random Forest Classifier predicts probability of user signup.
- One-hot encoding for categorical variables.
- Evaluated using classification metrics and ROC-AUC.

## 🎯 CTA Personalization
- If the model predicts a probability > 0.5 → “Start Now & Save 30%”
- Else → “Learn More About Us”

## 🧪 A/B Testing
- Users are randomly split into two groups:
  - **A (Control)**: Fixed CTA.
  - **B (Test)**: Personalized CTA.
- Signup outcomes are simulated.
- Two-proportion Z-test used to check statistical significance.

## 📈 Visualization
Bar chart showing conversion rates for both groups.

## 🛠️ How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
```

2. Run the script:
```bash
python ml_ab_test.py
```

## 📂 Files Included
- `ml_ab_test.py`: Main Python script
- `user_sessions.csv`: Sample dataset
- `README.md`: Project overview

## 🧠 Author
Created by an AI/ML and data science enthusiast. Feel free to modify and use this project in your own portfolio.
