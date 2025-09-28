import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns

from river import linear_model, preprocessing, metrics

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "C:/Users/Madhumitha/OneDrive/Documents/creditcard.csv"  # Kaggle dataset path
RANDOM_STATE = 42

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found. Download 'creditcard.csv' from Kaggle and place in the same folder.")
    exit()

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:", df.shape)

# Create datetime
base = datetime(2025, 1, 1)
df["txn_datetime"] = df["Time"].apply(lambda s: base + timedelta(seconds=float(s)))

# Synthetic card_id
np.random.seed(RANDOM_STATE)
df["card_id"] = np.random.choice([f"CARD_{i}" for i in range(1, 501)], size=len(df))

# -------------------------------
# STEP 2: BEHAVIORAL FEATURES
# -------------------------------
df["hour"] = df["txn_datetime"].dt.hour
df["day_of_week"] = df["txn_datetime"].dt.dayofweek
df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

profiles = df.groupby("card_id").agg(
    avg_amount=("Amount", "mean"),
    std_amount=("Amount", "std"),
    median_amount=("Amount", "median"),
    night_tx_ratio=("is_night", "mean"),
    tx_count=("Amount", "count")
).reset_index()

df = df.merge(profiles, on="card_id", how="left")

def anomaly(row):
    if pd.isna(row["std_amount"]) or row["std_amount"] == 0:
        return abs(row["Amount"] - row["median_amount"]) / (1 + row["median_amount"])
    return abs(row["Amount"] - row["avg_amount"]) / (row["std_amount"] + 1e-9)

df["behavioral_anomaly"] = df.apply(anomaly, axis=1).clip(0, 10) / 10

# -------------------------------
# STEP 3: RANDOM FOREST BASELINE
# -------------------------------
features = [c for c in df.columns if c.startswith("V")] + ["Amount", "hour", "day_of_week", "is_night"]
X = df[features].fillna(0)
y = df["Class"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")
rf.fit(X_train, y_train)

y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"\n✅ RandomForest ROC AUC: {auc:.4f}")
print(classification_report(y_test, (y_proba >= 0.5).astype(int)))

df.loc[idx_test, "rf_score"] = y_proba

# Confusion Matrix
cm = confusion_matrix(y_test, (y_proba >= 0.5).astype(int))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
plt.title("Confusion Matrix (RandomForest)")
plt.show()

# -------------------------------
# STEP 4: REAL-TIME ONLINE LEARNING
# -------------------------------
print("\n⚡ Running Online Learning with River...")

online_model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.ROCAUC()
risk_scores = []

for i, row in df.head(5000).iterrows():
    x = {
        "Amount": row["Amount"],
        "hour": row["hour"],
        "day_of_week": row["day_of_week"],
        "is_night": row["is_night"],
        "behavioral_anomaly": row["behavioral_anomaly"]
    }
    y_true = row["Class"]
    
    y_pred = online_model.predict_proba_one(x).get(True, 0.0)
    risk_scores.append(y_pred)
    
    online_model.learn_one(x, y_true)
    metric.update(y_true, y_pred)

df.loc[df.index[:5000], "online_score"] = risk_scores
print("✅ Online learning AUC (first 5000 txns):", metric.get())

# -------------------------------
# STEP 5: RISK SCORE + FRAUD CARDS
# -------------------------------
df["risk_score"] = 0.6 * df["rf_score"].fillna(0) + 0.4 * df["behavioral_anomaly"]

# Detect risky cards
fraud_cards = df[df["Class"]==1].groupby("card_id")["risk_score"].mean().sort_values(ascending=False).head(5)
print("\n🚨 Fraud Detected Cards:")
print(fraud_cards)

# -------------------------------
# STEP 6: PLOTS
# -------------------------------
print("\n📊 Plotting Fraud Risk Analysis...")

# Risk Distribution
plt.figure(figsize=(7,5))
plt.hist(df[df["Class"]==0]["risk_score"], bins=50, alpha=0.6, label="Legit")
plt.hist(df[df["Class"]==1]["risk_score"], bins=50, alpha=0.6, label="Fraud")
plt.title("Fraud vs Legit Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Count")
plt.legend()
plt.show()

# Fraudulent cards - risk scores
fraud_cards.plot(kind="bar", figsize=(8,5), color="red")
plt.title("Top 5 Fraudulent Cards by Avg Risk Score")
plt.ylabel("Avg Risk Score")
plt.show()

# Transaction timeline for fraud cards
plt.figure(figsize=(10,5))
for card in fraud_cards.index:
    sub = df[(df["card_id"]==card)].sort_values("txn_datetime")
    plt.plot(sub["txn_datetime"], sub["risk_score"], marker="o", label=card)
plt.title("Fraudulent Card Risk Evolution Over Time")
plt.xlabel("Time")
plt.ylabel("Risk Score")
plt.legend()
plt.show()

# Fraud amount analysis
plt.figure(figsize=(7,5))
plt.boxplot([df[df["Class"]==0]["Amount"], df[df["Class"]==1]["Amount"]], labels=["Legit","Fraud"])
plt.title("Transaction Amount Comparison")
plt.ylabel("Amount")
plt.show()

print("\n📂 Results saved to 'transactions_with_risk.csv'")
df.to_csv("transactions_with_risk.csv", index=False)
