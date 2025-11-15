print("Running 'test.py'...")

# importing packages
import requests

url = 'http://localhost:9696/predict'

# example applicant to predict
applicant = {
    'age': 52,
    'income': 95155,
    'loan_amount': 150365,
    'credit_score': 656,
    'months_employed': 3,
    'num_credit_lines': 4,
    'interest_rate': 18.54,
    'dti_ratio': 0.63
}

print("Applicant:", applicant)

response = requests.post(url, json=applicant)

predictions = response.json()

print("Applicant default probability:", round(predictions['default_probability'], 3))

if predictions['default']:
    print("Applicant predicted to default on loan, do not approve.")
else:
    print("Applicant not predicted to default on loan.")