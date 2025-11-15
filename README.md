# Loan Applicant Default Prediction

#### A FastAPI-based web service that uses a machine learning model to predict whether a bank loan applicant will default on repaying a loan.

## Overview

A key problem faced by every financial institution offering loans to private individuals is deciding whether the applicant is likely to pay off the loan amount, or whether the applicant is instead likely to default on the loan. Offering as many loans as possible to good applicants is important for maximising profits earned through interest, however this must be balanced against the need to avoid offering loans to applicants who will not pay the loan amount back; the opportunity cost of denying a loan to a good applicant is significantly less than the financial burden placed on the lending institution in the event of a loanee defaulting.

One solution to this problem is to leverage historical data concerning previous loanees, including on whether these loanees paid off the loan or defaulted, to develop prediction models to predict whether an applicant is likely to default on a proposed loan or not, then use these models to decide whether or not to grant the loan.

In this project, machine-learning techniques are used to develop a binary classification model, to predict whether or not a loan applicant is going to default on a loan.

### Caveat Lector

The primary objective of this midterm project, as far as concerns the author, was to practice implementing the steps of the "machine learning lifecycle", from data processing to model selection to model deployment, rather than necessarily doing everything possible to maximise final model performance (we refer the reader to the conclusion for some suggestions for how better performing models could be obtained).

## Data Source

The data was sourced from the [**`Loan Default Prediction Dataset`**](https://www.kaggle.com/datasets/nikhil1e9/loan-default) on Kaggle. The raw data can also be found as the file [**`data/raw_data.csv`**](data/raw_data.csv).

## Prerequisites

In order to run this project on your local system, the following need to be installed:

- Python 3.13 or above
- [**`uv`**](https://github.com/astral-sh/uv), for virtual environment management
- Docker

## Prediction Service

### Input Features

The prediction service takes as input the following features:

| Feature Name | Description |
|------------|-------------|
| `age` | The age of the applicant (`int`) |
| `income` | The annual income of the applicant, in USD (`int`) |
| `loan_amount` | The amount of money to be borrowed, in USD (`int`) |
| `credit_score` | The credit score of the applicant (`int`) |
| `months_employed` | The number of months the applicant has been deployed (`int`) |
| `num_credit_lines` | The number of credit lines the applicant has open (`int`) |
| `interest_rate` | The interest rate of the loan (`float`) |
| `dti_ratio` | The debt-to-income ratio of the applicant (`float` in range $0 \leq r \leq 1$) |

### Output

The prediction service outputs the following:

| Output | Description |
|------------|-------------|
| `default_probability` | The probability of the applicant defaulting on the loan, as computed by the model (`float`) |
| `default` | `= bool(default_probability >= 0.5)` (`bool`) |

### Docker Build/Run Instructions

 1. In a terminal, execute the following commands:

```bash
docker pull ball4772/midterm-predict-default:latest
docker run -p 9696:9696 ball4772/midterm-predict-default:latest
```

2. In a separate terminal, use `curl` to send requests to the prediction service, as follows:

```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": AGE,
           "income": INCOME,
           "loan_amount": LOAN AMOUNT,
           "credit_score": CREDIT SCORE,
           "months_employed": MONTHS EMPLOYED,
           "num_credit_lines": NUMBER OF CREDIT LINES,
           "interest_rate": INTEREST RATE,
           "dti_ratio": DTI RATIO
         }'
```

3. To end the prediction service, in the terminal where the service is running press `Ctrl + C`.

### Example Request

Input:

```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 52,
           "income": 95155,
           "loan_amount": 150365,
           "credit_score": 656,
           "months_employed": 3,
           "num_credit_lines": 4,
           "interest_rate": 18.54,
           "dti_ratio": 0.63
         }'
```

Output:

```bash
{"default_probability":0.585155897641126,"default":true}
```

## Local System Setup

### Cloning the Repository

1. Open Terminal (or Command Prompt in Windows), and change the current directory to the location where you want the cloned directory to be located.

2. Execute the following commands:

```bash
git clone https://github.com/gecooper01/ML_Zoomcamp_25_Midterm

cd ML_Zoomcamp_25_Midterm
```
### Initialising the Virtual Enviornment (for locally compiling/running [**`notebook.ipynb`**](notebook.ipynb), [**`train.py`**](train.py) and [**`predict.py`**](predict.py))

3. Create and activate the virtual environment:

```bash
uv venv 
source .venv/bin/activate # with Windows: .venv\Scripts\activate
```

4. Install dependencies:

```bash
uv pip sync
```

## Local System Testing

### Training/Testing Data Creation and Model Training

1. In a terminal, execute the following command:

```bash
uv run python train.py
```

Output:

```bash
Running 'train.py'...
Saved training data to '/data' as 'training_data.csv'
Saved testing data to '/data' as 'testing_data.csv'
Saved trained model as 'model.bin'
```

### Testing the Prediction Service (without using Docker)

2. Run the prediction service:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

3. In a separate terminal instance, execute the following command:

```bash
python test.py
```

Output:

```bash
Running 'test.py'...
Applicant: {'age': 52, 'income': 95155, 'loan_amount': 150365, 'credit_score': 656, 'months_employed': 3, 'num_credit_lines': 4, 'interest_rate': 18.54, 'dti_ratio': 0.63}
Applicant default probability: 0.585
Applicant predicted to default on loan, do not approve.
```

4. End the prediction service by pressing `Ctrl + C`.

### Testing the Prediction Service using Docker

5. In a terminal, pull and run the Docker container:

```bash
docker pull ball4772/midterm-predict-default:latest
docker run -p 9696:9696 ball4772/midterm-predict-default:latest
```

6. In a separate terminal instance, execute the following command:

```bash
python test.py
```

Output:

```bash
Running 'test.py'...
Applicant: {'age': 52, 'income': 95155, 'loan_amount': 150365, 'credit_score': 656, 'months_employed': 3, 'num_credit_lines': 4, 'interest_rate': 18.54, 'dti_ratio': 0.63}
Applicant default probability: 0.585
Applicant predicted to default on loan, do not approve.
```

7. End the prediction service by pressing `Ctrl + C`.

## Repository Structure and Files

### Project Structure

```markdown
.
├── data/
│   ├── `data_source.txt`
│   ├── `raw_data.csv`
│   ├── `testing_data.csv`
│   └── `training_data.csv`
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── model.bin
├── notebook.ipynb
├── predict.py
├── pyproject.toml
├── README.md
├── test.py
├── train.py
└── uv.lock
```

### Notebook

The Jupyter notebook [**`notebook.ipynb`**](notebook.ipynb) contains the following:

- Data preparation
- Training/testing data splitting
- Exploratory data analysis
- Feature selection
- Model selection

### Source Code Files

This project contains the following code files:

- [**`train.py`**](train.py): Training the final model using the full training dataset, as used in [**`notebook.ipynb`**](notebook.ipynb), then saving the final model along with the training and testing datasets.

- [**`predict.py`**](predict.py): Code for serving the prediction service, as well as for using the trained final model to generate predictions, given an input.

- [**`test.py`**](test.py): Example request to the prediction service - requires the prediction service to be running in a separate terminal instance.

- [**`Dockerfile`**](Dockerfile): Docker configuration file for containerising the prediction service.

- [**`pyproject.toml`**](pyproject.toml): `uv` configuration file for this project.

## Summary of Data Preparation and EDA

### Complete Dataset Description

We refer to the description present on [**`Kaggle`**](https://www.kaggle.com/datasets/nikhil1e9/loan-default) for the original dataset.

- In both the original and processed datasets, a value of `0` for `default` denotes a loanee who did not default on their loan, and a value of `1` denotes a loanee who did default.

### Data Preparation Steps

- The `LoanID` column was deleted, due to being redundant.
- A check was performed for any missing or duplicated entries (neither kind of entry was present).
- Column names and categorical entries were formatted to be all lower case and free from punctuation, except for underscores.
- An 80%/20% train/test split was conducted, with the same random state as used for the train/test split in [**`train.py`**](train.py).

### EDA/Feature Selection Summary

- EDA was performed only on the training dataset, to ensure any resulting feature selection was not dependent on the testing dataset.
- Value counts for the target feature, `default`, were taken.
    - Class imbalance for `default` was identified: 88.4% of the entries were in class `0` (no default) and 11.6% of the entries were in class `1` (default).
- For each of the categorical features, value counts were computed, and the average default rate for each category was compared against the global average default rate. Mutual information (MI) between each of the categorical features and `default` was also computed.
    - Due to all of the categorical variables having MI at most 0.0011, all of the categorical features were omitted during feature selection.
- For each of the numerical features, histograms were created to plot their distributions. The correlation between each of the numerical features and `default` was also computed.
    - Due to having a minimal absolute value for the correlation with `default` when compared with all other numerical features, the feature `loan_term` was omitted during feature selection.

## Model Selection

### Baseline Model

For comparing against our later trained models, the following baseline model was trained (after performing a separate testing/validation split):

```python
baseline_model = LogisticRegression(
        random_state=7
    )
```

This model produced the following classification report upon evaluation using the validation data:

```python
Classification Report:
              precision    recall  f1-score   support

           0       0.89      1.00      0.94     36153
           1       0.66      0.02      0.04      4703

    accuracy                           0.89     40856
   macro avg       0.77      0.51      0.49     40856
weighted avg       0.86      0.89      0.84     40856

F2 score: 0.028194993412384718
```

- Recall that the $F_{\beta}$-score is defined by the formula

    $$ F_{\beta} = \frac{(1 + \beta^2) \cdot \mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}} = \frac{(1 + \beta^2) \cdot \mathrm{TP}}{(1 + \beta^2) \cdot \mathrm{TP} + \beta^2 \cdot \mathrm{TN} + \mathrm{FP}}, \quad \beta \in \mathbb{R}^{>0}. $$

This is consistent with a model predicting almost all applicants to not default.

### Model Types Considered

1. Logistic regression
2. Decision tree classifier
3. XGBoost classifier

Originally, a random forest classifier was also considered, however this was later omitted due to this model type being significantly slower to train than the other types considered.

### Pipeline Assembly

For each model type considered, the following pipeline was formed, consisting of a preprocessing stage to normalise and rescale each of the numerical features, and a prediction stage:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# features selected after feature selection
features = ['age', 'income', 'loan_amount', 'credit_score', 'months_employed', 'num_credit_lines', 'interest_rate', 'dti_ratio']

# preprocessor
preprocess = ColumnTransformer(
    transformers=[
        ('standard_scaler', StandardScaler(), features),
    ]
)

# pipeline
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)) # for example, model = DecisionTreeClassifier(random_state=42)
```

### Hyperparameter Tuning

For each model type, hyperparameter tuning was performed using 5-fold stratified (to take into account class imbalance) cross validation. To save resources, `RandomizedSearchCV()` was used to sample 50 parameter choices from each specified parameter distribution, totalling 250 training runs per model type.

The best choice of hyperparameters for each model type was determined by maximising the average $F_2$-score across the five folds; this was chosen due to false negatives being significantly more problematic than false positives (in other words, we prioritise maximising recall over maximising precision, whilst still wanting to have a model which is not a dummy model predicting most/all applicants will default).

### Final Model Selection

After performing hyperparameter tuning, it remained to select from the three models below:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# linear regression
lr_model = LogisticRegression(
        random_state=7,
        solver='sag',
        max_iter=50,
        class_weight='balanced',
        C=0.001
    )

# decision tree classifier
dt_model = DecisionTreeClassifier(
        random_state=7,
        min_samples_split=5,
        min_samples_leaf=8,
        max_features=None,
        max_depth=5,
        criterion='gini',
        class_weight='balanced'
    )

# XGBoost classifier
xgb_model = XGBClassifier(
        random_state=7,
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='logloss',
        subsample=0.8,
        scale_pos_weight=5,
        n_estimators=100,
        min_child_weight=5,
        max_depth=3,
        learning_rate=0.2,
        colsample_bytree=0.8
    )
```

A separate 5-fold stratified cross-validation was performed to compare these models, from which classification reports were calculated, to compare the mean $F_1$, $F_2$, precision and recall scores across the five folds. The classification reports produced are as follows:

#### Logistic Regression

| Class | Metric     | Mean      | Std       |
|:------|:-----------|:----------|:----------|
| 0     | f1         | 0.779223  | 0.002085  |
|       | f2         | 0.706351  | 0.002495  |
|       | precision  | 0.941032  | 0.001632  |
|       | recall     | 0.664898  | 0.002696  |
| 1     | f1         | 0.322170  | 0.004138  |
|       | f2         | 0.471522  | 0.006029  |
|       | precision  | 0.210858  | 0.002763  |
|       | recall     | 0.682438  | 0.009053  |
| macro | f1         | 0.550697  | 0.002911  |
|       | f2         | 0.588937  | 0.003733  |
|       | precision  | 0.575945  | 0.002163  |
|       | recall     | 0.673668  | 0.004925  |

#### Decision Tree Classifier

| Class | Metric     | Mean      | Std       |
|:------|:-----------|:----------|:----------|
| 0     | f1         | 0.778061  | 0.018908  |
|       | f2         | 0.706707  | 0.025105  |
|       | precision  | 0.936171  | 0.001478  |
|       | recall     | 0.666050  | 0.027888  |
| 1     | f1         | 0.311702  | 0.008391  |
|       | f2         | 0.454046  | 0.004353  |
|       | precision  | 0.204839  | 0.008825  |
|       | recall     | 0.653696  | 0.020505  |
| macro | f1         | 0.544881  | 0.013481  |
|       | f2         | 0.580377  | 0.013328  |
|       | precision  | 0.570505  | 0.004162  |
|       | recall     | 0.659873  | 0.005376  |

#### XGBoost Classifier

| Class | Metric     | Mean      | Std       |
|:------|:-----------|:----------|:----------|
| 0     | f1         | 0.870537  | 0.001426  |
|       | f2         | 0.839727  | 0.001897  |
|       | precision  | 0.927240  | 0.001012  |
|       | recall     | 0.820372  | 0.002203  |
| 1     | f1         | 0.353908  | 0.004730  |
|       | f2         | 0.433233  | 0.005777  |
|       | precision  | 0.271164  | 0.003833  |
|       | recall     | 0.509349  | 0.007110  |
| macro | f1         | 0.612223  | 0.002937  |
|       | f2         | 0.636480  | 0.003365  |
|       | precision  | 0.599202  | 0.002373  |
|       | recall     | 0.664860  | 0.003816  |

Based on having the best class `1` recall and $F_2$-score, the logistic regression model was selected as the final model.

## Conclusion

The final model has a class `1` average recall of 68.2%, meaning that the model was able to (on average) identify 68.2% of actual defaultees as applicants who would default. On the other hand, the final model has a class `1` average precision of 21.0%, meaning that when the model predicted an applicant would default, only 21.0% of those applicants actually ended up defaulting. This is similarly reflected in the class `0` average recall being 66.5%.

As a starting point, this performance is not bad, however our final model is far from optimal; to give one recent example from the literature, an XGBoost model for loan default prediction achieved a class `1` recall of 80.2%, precison of 81.6% and $F_1$-score of 80.8%, a significantly better performance compared to our model [[1]](#1).

### Suggestions for Further Improvements

The following list contains some suggestions for obtaining better performing models:

1. Consider a wider class of classifiers, including those suited to situations with class imbalance (such as the balanced bagging classifier from the `imblearn` library).
2. Implement resampling techniques, such as implementing `imblearn`'s `RandomOverSampler` and/or `RandomUnderSampler`, again to overcome class imbalance.
3. Perform a more thorough implementation of hyperparameter tuning, including using a wider range of parameters to choose from, and performing more training runs for each model type.
4. Re-introduce into the training dataset some or all of the categorical features (appropriately one-hot encoded).
5. Perform a threshold analysis to select an optimal threshold (in this case, for deciding when to predict that an applicant will default).
6. Use multiple metrics to monitor model performance during hyperparameter tuning, as opposed to only a single metric (in our case, the $F_2$-score), to prevent class `1` precision from falling too low.

## References

<a id="1">[1]</a> Zhang X., Zhang T., Hou L., Liu X., Guo Z., Tian Y., Liu Y. *Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management.* Systems **2025**, 13, 581. [`https://doi.org/10.3390/systems13070581`](https://doi.org/10.3390/systems13070581)