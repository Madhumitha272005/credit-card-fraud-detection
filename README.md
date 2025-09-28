💳 credit-card-fraud-detection
This project implements a comprehensive credit card fraud detection system entirely in Python, combining batch machine learning and real-time online learning. The system leverages Random Forest for batch predictions, behavioral anomaly detection, and River's online learning (Logistic Regression with StandardScaler) to detect fraudulent transactions with high accuracy and adaptability.

The approach includes feature engineering based on transaction time, card behavior, and statistical metrics, allowing the model to detect anomalous spending patterns. Risk scores are calculated by combining model predictions with behavioral anomalies, identifying high-risk cards and transactions.

Key Features:
Batch ML Model: Random Forest trained on historical data for robust fraud prediction.
Behavioral Analysis: Features like transaction hour, day, night activity, average/median amount per card.
Online Learning: Real-time fraud detection using River’s Logistic Regression and StandardScaler.
Risk Scoring: Combines Random Forest predictions and behavioral anomaly metrics to flag risky transactions/cards.
Visualizations: Risk distribution, top fraudulent cards, transaction timelines, and transaction amount comparisons.
Output: Generates transactions_with_risk.csv with fraud scores and risk metrics.

Tech Stack:
Programming Language: Python
Libraries: pandas, numpy, scikit-learn, river, matplotlib, seaborn
Dataset: Kaggle Credit Card Fraud Detection
Environment: Python IDLE

Credit Card Fraud Detection Working:
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

ML ALGORITHM USED INT HIS PROJECT:
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

	​



