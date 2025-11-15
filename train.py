print("Running 'train.py'...")

# importing packages
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

# loading the dataset
file = 'data/raw_data.csv'
df = pd.read_csv(file)

# removing the 'LoanID' column, as this is not needed
del df['LoanID']

# formatting column names
column_name_map = {
    "Age": "age",
    "Income": "income",
    "LoanAmount": "loan_amount",
    "CreditScore": "credit_score",
    "MonthsEmployed": "months_employed",
    "NumCreditLines": "num_credit_lines",
    "InterestRate": "interest_rate",
    "LoanTerm": "loan_term",
    "DTIRatio": "dti_ratio",
    "Education": 'education',
    "EmploymentType": "employment_type",
    "MaritalStatus": "marital_status",
    "HasMortgage": "has_mortgage",
    "HasDependents": "has_dependents",
    "LoanPurpose": "loan_purpose",
    "HasCoSigner": "has_cosigner",
    "Default": "default"
}

df = df.rename(columns=column_name_map)

# deleting features identified for removal in feature selection
for col in ['loan_term', 'education', 'employment_type', 'marital_status', 'has_mortgage', 'has_dependents', 'loan_purpose', 'has_cosigner']:
    del df[col]

# feature names
features = ['age', 'income', 'loan_amount', 'credit_score', 'months_employed', 'num_credit_lines', 'interest_rate', 'dti_ratio']

# train/test split (same random state as in notebook)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=5)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# saving training and testing data
df_train.to_csv('data/training_data.csv')
print("Saved training data to '/data' as 'training_data.csv'")

df_test.to_csv('data/testing_data.csv')
print("Saved testing data to '/data' as 'testing_data.csv'")

# splitting off the target feature from the training dataframe
y_train = df_train['default'].values

del df_train['default']

# preprocessor
preprocess = ColumnTransformer(
    transformers=[
        ('standard_scaler', StandardScaler(), features),
    ]
)

# final model architecture
model = LogisticRegression(
        random_state=7,
        solver='sag',
        max_iter=50,
        class_weight='balanced',
        C=0.001
    )

# combining model and preprocessor into a pipeline
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

# training the final model
pipeline.fit(df_train, y_train)

# saving the final model
with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)

print("Saved trained model as 'model.bin'")