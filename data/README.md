# Data

Raw data is **not committed** (see `.gitignore`). Get it with either:

```bash
churn-download          # needs Kaggle credentials
```

or download manually from
<https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset> and put these two files in `data/raw/`:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

| Column | Type | Notes |
|---|---|---|
| CustomerID | float | identifier, dropped before modelling |
| Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction | numeric | |
| Gender, Subscription Type, Contract Length | categorical | one-hot encoded |
| Churn | 0/1 | target |
