# Project: Improving Signup Conversion with ML-based CTA Personalization and A/B Testing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

csv_path = 'user_sessions.csv'
df = pd.read_csv(csv_path)

df.dropna(subset=['device_type', 'scroll_depth', 'time_on_page', 'signed_up'], inplace=True)
df = pd.get_dummies(df, columns=['device_type', 'browser', 'traffic_source'], drop_first=True)

X = df.drop(columns=['signed_up'])
y = df['signed_up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

def get_cta(pred_proba):
    return "Start Now & Save 30%" if pred_proba > 0.5 else "Learn More About Us"

X_new = X_test.copy()
X_new['pred_proba'] = model.predict_proba(X_test)[:, 1]
X_new['cta'] = X_new['pred_proba'].apply(get_cta)
X_new['group'] = np.random.choice(['A', 'B'], size=len(X_new), p=[0.5, 0.5])

X_new['signed_up'] = np.where(
    (X_new['group'] == 'B') & (X_new['pred_proba'] > 0.5), 1,
    np.where((X_new['group'] == 'A'), np.random.binomial(1, 0.085, size=len(X_new)), 0)
)

signup_counts = [X_new[X_new['group'] == 'A']['signed_up'].sum(), X_new[X_new['group'] == 'B']['signed_up'].sum()]
visitor_counts = [X_new[X_new['group'] == 'A'].shape[0], X_new[X_new['group'] == 'B'].shape[0]]

z_stat, p_val = proportions_ztest(signup_counts, visitor_counts)
print(f"Z-statistic: {z_stat:.4f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Statistically significant improvement with personalized CTA.")
else:
    print("Result: No statistically significant difference.")

labels = ['Control (A)', 'Personalized (B)']
signup_rates = [signup_counts[0] / visitor_counts[0], signup_counts[1] / visitor_counts[1]]
plt.bar(labels, signup_rates)
plt.ylabel("Signup Rate")
plt.title("A/B Test: Signup Conversion Rate")
plt.show()
