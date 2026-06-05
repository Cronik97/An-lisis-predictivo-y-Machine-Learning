import time
import pandas as pd
from pymongo import MongoClient
import joblib

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["predictivo_fallas"]
coleccion = db["sensores"]

# Cargar modelo entrenado (ruta relativa)
modelo = joblib.load("models/modelo_fallas.pkl")

print("✅ Sistema de predicción iniciado... monitoreando datos en tiempo real")

# Bucle infinito para revisar nuevos datos
while True:
    # Obtener el último registro insertado en MongoDB
    ultimo = coleccion.find().sort([("_id", -1)]).limit(1)
    for registro in ultimo:
        # Extraer variables predictoras
        X_nuevo = pd.DataFrame([{
            "temperatura": registro["temperatura"],
            "vibracion": registro["vibracion"],
            "voltaje": registro["voltaje"]
        }])

        # Realizar predicción y mapear a 0/1 donde 1 = falla
        raw = modelo.predict(X_nuevo)[0]
        model_name = modelo.__class__.__name__
        if model_name == "IsolationForest":
            prediccion = 1 if raw == -1 else 0
        else:
            try:
                prediccion = int(raw)
                prediccion = 1 if prediccion == 1 else 0
            except Exception:
                prediccion = 0

        if prediccion == 1:
            print(f"⚠️ ALERTA: Posible falla detectada en {registro['timestamp']}")
        else:
            print(f"✅ Operación normal en {registro['timestamp']}")

    # Esperar 5 segundos antes de revisar de nuevo
    time.sleep(5)
