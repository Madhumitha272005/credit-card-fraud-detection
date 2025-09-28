💳 credit-card-fraud-detection
This project implements a comprehensive credit card fraud detection system entirely in Python, combining batch machine learning and real-time online learning. The system leverages Random Forest for batch predictions, behavioral anomaly detection, and River's online learning (Logistic Regression with StandardScaler) to detect fraudulent transactions with high accuracy and adaptability.

The approach includes feature engineering based on transaction time, card behavior, and statistical metrics, allowing the model to detect anomalous spending patterns. Risk scores are calculated by combining model predictions with behavioral anomalies, identifying high-risk cards and transactions.

🎯 Key Features:
Batch ML Model: Random Forest trained on historical data for robust fraud prediction.
Behavioral Analysis: Features like transaction hour, day, night activity, average/median amount per card.
Online Learning: Real-time fraud detection using River’s Logistic Regression and StandardScaler.
Risk Scoring: Combines Random Forest predictions and behavioral anomaly metrics to flag risky transactions/cards.
Visualizations: Risk distribution, top fraudulent cards, transaction timelines, and transaction amount comparisons.
Output: Generates transactions_with_risk.csv with fraud scores and risk metrics.

🔧Tech Stack:
Programming Language: Python
Libraries: pandas, numpy, scikit-learn, river, matplotlib, seaborn
Dataset: Kaggle Credit Card Fraud Detection
Environment: Python IDLE

✅Credit Card Fraud Detection Working:
Credit Card Fraud Detection is a process that identifies unauthorized or suspicious transactions to protect users and financial institutions. In your project, the system works through a combination of machine learning models, behavioral analysis, and risk scoring.

Step 1: Data Preparation
The system starts by loading transaction data (e.g., creditcard.csv).
Each transaction includes features like transaction amount, time, and anonymized PCA components (V1–V28).
Additional behavioral features are generated, such as:
   Hour of transaction
   Day of week 
   Night transaction flag
   Card-based statistics like average, median, and standard deviation of amounts

Step 2: Feature Engineering & Behavioral Analysis
The system calculates behavioral anomalies, which measure how unusual a transaction is compared to the typical behavior of that card (e.g., spending far above the average).
These features help the model detect suspicious patterns that are not evident from single transactions.

Step 3: Machine Learning Models
A Random Forest Classifier is trained on historical transaction data to predict whether a transaction is fraudulent.
The model outputs a probability score for each transaction, indicating the likelihood of fraud.
For real-time adaptability, an online learning model (River’s Logistic Regression) continuously updates as new transactions arrive, allowing the system to detect emerging fraud patterns.

Step 4: Risk Scoring
The final risk score combines:
Random Forest predictions (batch model)
Behavioral anomaly scores
This combined score identifies high-risk transactions and potentially fraudulent cards.

Step 5: Detection and Visualization
Transactions above a certain risk threshold are flagged as potential fraud.
Visualizations like risk score distributions, top fraudulent cards, timelines, and transaction amount comparisons help understand patterns and evaluate model performance.

Step 6: Deployment
Trained models are saved as .pkl files for easy integration into real-time applications or web interfaces.
The system can continuously monitor transactions and adapt to new fraud patterns.

🧮 ML algorithm used in this project :
1. Random Forest Classifier
Type: Ensemble Learning (Classification)
Formula (Majority Voting):  y^​=mode{T1​(x),T2​(x),...,Tn​(x)}

2. Logistic Regression (Online Learning)
Type: Linear Model (Classification)
Formula (Sigmoid function): p=1/1+e−(β0​+∑i=1n​βi​xi​)

3. Behavioral Anomaly Score (Custom Metric)
While not a standard ML algorithm, your project computes a behavioral anomaly score to complement ML predictions: behavioral_anomaly=
∣Amount−Avg/Median Amount per Card∣/Std Amount + small constant
Helps detect unusual spending behavior for each card.
Normalized to [0,1] and combined with Random Forest probability to compute the final risk score: risk_score=0.6×rf_score+0.4×behavioral_anomaly

✅ ouput explanation:
1. Dataset Loaded
   ✅ Dataset loaded: (284807, 31)
My dataset contains 284,807 transactions with 31 columns (features + class label).
This includes PCA components (V1–V28), Amount, Time, and the class label (Class: 0 = Legit, 1 = Fraud).​

2. Random Forest Model
   ✅ RandomForest ROC AUC: 0.9276
ROC-AUC = 0.9276 → The model has very good discrimination between legitimate and fraudulent transactions.
Classification Report:
Class	   Precision  Recall  F1-score	Support
0 (Legit)	1.00	   1.00	  1.00	    85,295
1 (Fraud)	0.97	   0.70	  0.82	    148
Legit transactions are almost perfectly classified.
Fraud transactions:
Precision = 0.97 → 97% of predicted frauds were actually fraud.
Recall = 0.70 → 70% of actual frauds were detected.
F1-score = 0.82 → Good balance considering dataset is highly imbalanced.
Accuracy ≈ 1.00 → Very high due to class imbalance (most transactions are legit).
Key takeaway: Random Forest is effective but may miss ~30% of fraud cases due to imbalance.

3. Online Learning (River)
   ⚡ Running Online Learning with River...
✅ Online learning AUC (first 5000 txns): 0.7166
Online Logistic Regression trained incrementally on first 5,000 transactions.
AUC = 0.7166 → Moderate predictive performance.
Slightly lower than batch Random Forest because:
Only first 5,000 transactions were used.
Online learning adapts gradually; requires more data to reach peak performance.
Purpose: Detect emerging fraud patterns in real-time as new transactions arrive.

4. Fraud Detected Cards
   🚨 Fraud Detected Cards:
 card_id
 CARD_472    0.616078
 CARD_228    0.613106
 CARD_45     0.604736
 CARD_375    0.602671
 CARD_92     0.602401
The risk score combines Random Forest predictions and behavioral anomalies:
risk_score=0.6×rf_score+0.4×behavioral_anomaly
These 5 cards have the highest average risk score → most likely to be involved in fraudulent activities.
This helps investigators prioritize monitoring or blocking suspicious cards.

5. Graph Analysis
 a) Risk Score Distribution
Histogram comparing legit vs fraud transactions:
Legit transactions cluster at low risk scores.
Fraud transactions are concentrated at higher risk scores.
Insight: Risk score successfully separates most fraud from legit transactions.

 b) Top 5 Fraudulent Cards (Bar Chart)
Shows highest-risk cards by average risk score.
Observation: CARD_472 is the riskiest, followed by CARD_228, CARD_45, etc.
Useful for visual prioritization of fraudulent accounts.

 c) Fraudulent Card Timeline (Line Plot)
Plots risk score over time for each of the top 5 fraud cards.
Observation:
Sudden spikes indicate unusual transactions.
Continuous monitoring allows real-time detection of fraud trends.

 d) Transaction Amount Comparison (Boxplot)
Compares legit vs fraud transaction amounts.
Observation: Fraudulent transactions often have higher amounts or outliers.
Highlights anomalous spending behavior that supports risk scoring.


