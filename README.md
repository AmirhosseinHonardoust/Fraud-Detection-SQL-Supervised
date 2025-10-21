# Fraud Detection [SQL + Python (Supervised)]

Predict fraudulent transactions using **SQL (SQLite)** for feature engineering and **Python** with Logistic Regression for supervised classification.

---

## Overview

This project extends the unsupervised version by introducing **labeled data** and **supervised learning**.  
It demonstrates a complete fraud prediction pipeline, from SQL feature generation to model training, evaluation, and visualization.

---

## Workflow

1. **Load labeled data into SQLite**
2. **Run SQL feature engineering**
   - Compute per-user and daily transaction statistics
3. **Train Logistic Regression model**
   - Input: engineered SQL features
   - Output: fraud probability for each transaction
4. **Evaluate model performance**
   - AUC, Precision, Recall, F1-score
5. **Visualize ROC curve**

---

## Project Structure

```
fraud-detection-sql-supervised/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ transactions_labeled.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ train_supervised.py
│  └─ utils.py
└─ outputs/
   ├─ metrics.json
   ├─ fraud_scores.csv
   ├─ fraud_summary.csv
   └─ charts/
       └─ roc_curve.png
```

---

## Dataset Schema

| Column | Description |
|---------|--------------|
| tx_id | Transaction ID |
| user_id | Unique user identifier |
| date | Transaction date |
| region | User region |
| merchant | Merchant name |
| amount | Transaction amount |
| label | 1 = Fraudulent, 0 = Legitimate |

---

## SQL Feature Engineering

Feature generation reuses the same structure as the unsupervised project.

```sql
CREATE TEMP VIEW user_stats AS
SELECT user_id, COUNT(*) AS tx_count, AVG(amount) AS avg_amount, SUM(amount) AS total_amount
FROM transactions
GROUP BY user_id;

CREATE TEMP VIEW daily_user AS
SELECT user_id, date, COUNT(*) AS daily_tx, SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;

SELECT t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
       us.tx_count, us.avg_amount, us.total_amount,
       COALESCE(du.daily_tx, 0) AS daily_tx,
       COALESCE(du.daily_amount, 0.0) AS daily_amount,
       t.label
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date;
```

---

## Machine Learning

Model: **Logistic Regression**

- Trained on labeled transaction data  
- Balanced class weights for rare fraud cases  
- Evaluated using ROC AUC, precision, recall, and F1-score  
- Generates probability scores (`fraud_proba`) for each transaction

---

## Visualization

### ROC Curve
<img width="900" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/db3669b0-0372-47d8-a3bf-08584dd9e94b" />

The ROC curve shows the trade-off between true positive rate (recall) and false positive rate.  
A curve closer to the top-left corner indicates stronger predictive performance.

---

## Tools & Libraries

| Tool | Purpose |
|------|----------|
| **SQLite** | Data storage and feature generation |
| **Python** | ML training and evaluation |
| **pandas** | Data handling |
| **scikit-learn** | Model building and metrics |
| **matplotlib** | Visualization |

---

## Usage

### Load Data into SQLite
```bash
python src/create_db.py --csv data/transactions_labeled.csv --db fraud.db
```

### Train and Evaluate Model
```bash
python src/train_supervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

---

## Outputs

| File | Description |
|------|--------------|
| `metrics.json` | Model performance metrics |
| `fraud_scores.csv` | Ranked transactions with fraud probability |
| `fraud_summary.csv` | Aggregated user-level fraud summary |
| `roc_curve.png` | ROC curve visualization |

---

## Conclusion

This project demonstrates a complete **supervised fraud detection workflow** using SQL and Python.  
It combines data engineering, model training, and evaluation into a single reproducible pipeline suitable for production-ready analytics and portfolio demonstration.
