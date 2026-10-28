# Customer Churn Prediction

A logistic regression model that predicts which telecommunications customers are likely to leave, built as the culminating project for IN503 Introduction to Machine Learning at Purdue Global.

## Project Description

Replacing a customer costs far more than keeping one, so identifying who is likely to leave before they do determines whether a retention budget is spent usefully. This project builds an end-to-end solution on the Telco Customer Churn dataset: 7,043 customer records covering demographics, subscribed services, and account details, with a binary target indicating whether the customer left.

The work runs the full sequence from exploratory analysis through preprocessing, feature engineering, hyperparameter tuning, evaluation, coefficient interpretation, and a saved model with a working prediction function.

Interpretability drove the choice of algorithm. Gradient boosting would likely score somewhat higher on this data, but logistic regression returns coefficients that convert into odds ratios a retention team can act on. A model that predicts well and explains nothing gives the business no direction.

## Repository Contents

| File | Description |
|---|---|
| `Holmes_Danielle_IN503_Unit10_Assignment.ipynb` | Full analysis notebook with code, visualizations, and written interpretation |
| `customer_churn_model.pkl` | Trained pipeline including all preprocessing steps |
| `requirements.txt` | Python packages needed to run the notebook |
| `README.md` | This file |

## How to Run the Code

### Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com) and upload the notebook.
2. Run all cells: **Runtime → Run all**.
3. The dataset downloads automatically through `kagglehub`; no manual download is needed.
4. `customer_churn_model.pkl` is written to the session directory and can be downloaded from the file browser.

### Local Environment

```bash
git clone https://github.com/DaniH-PG/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Holmes_Danielle_IN503_Unit10_Assignment.ipynb
```

### Using the Saved Model

The pickled file contains the entire pipeline, so raw customer records can be passed straight in without reproducing any preprocessing:

```python
import joblib
import pandas as pd

model = joblib.load('customer_churn_model.pkl')

customer = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 1, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 75.75, 'TotalCharges': 75.75, 'TotalServices': 2
}

probability = model.predict_proba(pd.DataFrame([customer]))[0][1]
print(f"Churn probability: {probability:.4f}")
```

`TotalServices` must be recomputed for any new record, counting how many of the nine service columns hold a value other than `No`, `No internet service`, or `No phone service`.

## Key Findings

**Data quality.** `TotalCharges` loads as text rather than a number because eleven rows contain a single space instead of a value, which a standard `isnull()` check does not catch. Every one of those rows has a tenure of zero, meaning they are new customers who have not been billed yet. The value is not missing through error. Removing them is defensible at this scale but quietly excludes the newest customers, who are the group most at risk of leaving.

**Contract type is the clearest driver.** Month-to-month customers churn at a substantially higher rate than those on one or two year agreements. A month-to-month customer faces no exit cost and can leave whenever a competitor makes a better offer.

**Fiber optic subscribers leave more than DSL subscribers.** This is counterintuitive for a premium product and points at either price or unmet reliability expectations. The company is losing its highest-paying customers.

**Tenure is strongly protective.** Churn concentrates heavily in the first months and thins considerably as tenure grows, which argues for focusing retention effort on onboarding rather than spreading it evenly across the base.

**Higher monthly charges track with churn**, overlapping with the fiber finding and raising the question of whether customers see the price as matching the value delivered.

**Engineered feature.** `TotalServices` counts how many of nine available services each customer subscribes to. Bundling changes the relationship: a customer with one service is buying a commodity and can switch easily, while a customer with six has to replace several things at once. That switching cost appears in no existing column, and it suggests a concrete intervention, which is offering an additional service instead of a discount.

**Model performance.** On the held-out test set of 1,407 customers, the tuned model (C = 100) reached 79.67% accuracy, compared with 73.42% for a baseline that always predicts no churn. For the churn class, precision was 63.17%, recall was 56.42% (211 of 374 churners caught), and F1 was 59.60%. ROC-AUC was 0.8348 on the test set, close to the 0.8462 cross-validated score, so the tuning did not overfit.

## Evaluation Approach

Accuracy alone is misleading on this target because the classes are imbalanced. A model predicting no churn for every customer would score the majority class rate while identifying nobody at risk, so the notebook reports accuracy against that baseline explicitly, alongside precision, recall, F1, a confusion matrix, and ROC-AUC.

Recall on the churn class carries the most weight for a retention use case, since a customer the model fails to flag receives no intervention at all. Hyperparameter tuning on `C` was scored with `roc_auc` rather than accuracy for the same reason: ranking quality is what a prioritized call list actually depends on.

## Recommendations

- Concentrate retention effort in the first months of the customer relationship
- Test contract conversion incentives for month-to-month customers, offering added value rather than price cuts to preserve margin
- Investigate the fiber optic churn rate as a product or pricing issue rather than a retention campaign target
- Use the model's probability output to rank the base and work down the list, rather than treating every flagged customer identically

## Limitations

The eleven removed records were all zero-tenure customers, the segment most exposed to churn. Coefficients describe association rather than causation, so a strong predictor is not automatically a lever that changes behavior. The model was trained on a single snapshot and will degrade as pricing and competition shift, so it needs scheduled retraining rather than waiting for someone to notice it has stopped working.

One caution on acting on the output: ranking customers by how likely they are to leave is not the same as ranking them by how much a retention offer would change their behavior. Some high-risk customers are leaving regardless, and spending on them is wasted. Measuring the actual effect of an intervention requires a holdout group that receives no offer.

## Dependencies

```
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn joblib
```

Developed on Python 3.10+ with scikit-learn 1.6.1. The saved model requires scikit-learn 1.6 or newer to load.

## Dataset

Telco Customer Churn, available on Kaggle as `blastchar/telco-customer-churn`. 7,043 records, 21 features, binary churn target.

## Author

Danielle Holmes
IN503 Introduction to Machine Learning, Purdue Global
