import time
import numpy as np
import pandas as pd
from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["predictivo_fallas"]
coleccion = db["sensores"]

print("✅ Simulación iniciada... generando datos cada 5 segundos")

while True:
    # Generar un registro simulado
    registro = {
        "temperatura": float(np.random.normal(70, 5)),
        "vibracion": float(np.random.normal(0.5, 0.1)),
        "voltaje": float(np.random.normal(220, 10)),
        "timestamp": pd.Timestamp.now()
    }

    # Insertar en MongoDB
    coleccion.insert_one(registro)

    print(f"📥 Registro insertado: {registro}")

    # Esperar 5 segundos antes de generar el siguiente
    time.sleep(5)
