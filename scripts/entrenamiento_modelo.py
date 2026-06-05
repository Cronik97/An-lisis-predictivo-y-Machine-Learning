import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["predictivo_fallas"]
coleccion = db["sensores"]

# Cargar datos desde MongoDB
datos = pd.DataFrame(list(coleccion.find()))
if datos.empty:
	print("⚠️ No hay datos en la colección para entrenar.")
	raise SystemExit(0)

# Seleccionar variables predictoras
X = datos[["temperatura", "vibracion", "voltaje"]]

if "falla" in datos.columns and datos["falla"].notna().sum() >= 10:
	# Entrenamiento supervisado si hay etiquetas suficientes
	y = datos["falla"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	modelo = RandomForestClassifier(n_estimators=100, random_state=42)
	modelo.fit(X_train, y_train)

	y_pred = modelo.predict(X_test)
	print("✅ Entrenamiento supervisado completado")
	print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
	print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
else:
	# Entrenamiento no supervisado: detectar anomalías (falla = anomalía)
	from sklearn.ensemble import IsolationForest
	modelo = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
	modelo.fit(X)
	print("✅ Entrenamiento no supervisado completado (IsolationForest)")
	print("ℹ️ El modelo detectará anomalías que se interpretan como fallas (1 = anomalía).")

# Guardar modelo entrenado
os.makedirs("models", exist_ok=True)
joblib.dump(modelo, os.path.join("models", "modelo_fallas.pkl"))

print("📦 Modelo guardado en models/modelo_fallas.pkl")
