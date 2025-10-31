import requests

# URL del servicio FastAPI
url = "http://localhost:9696/predict"

# Datos de entrada del cliente
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

# Enviar la solicitud POST al endpoint
response = requests.post(url, json=client)

# Obtener la respuesta en formato JSON
prediction = response.json()

# Mostrar resultado
print(prediction)

if prediction["churn"]:
    print("Client is likely to churn — send retention offer.")
else:
    print("Client is not likely to churn.")
