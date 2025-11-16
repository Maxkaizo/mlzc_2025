import requests

url = 'http://localhost:9696/predict'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    #'streamingtv': 'maybe', # it ws set to maybe to force an error
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85#,
    #'whatever': 31337 
}

response = requests.post(url, json=customer)

if response.status_code != 200:
    print('Request failed. Status:', response.status_code)
    print('Response:', response.json())
    raise SystemExit(1)

predictions = response.json()

if predictions['churn']:
    print('customer is likely to churn, send promo')
else:
    print('customer is not likely to churn')
