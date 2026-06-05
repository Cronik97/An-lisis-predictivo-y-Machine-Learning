# main.py
import tkinter as tk
from tkinter import scrolledtext, filedialog
from threading import Thread, Lock, Event
import time
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from collections import deque

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --------------------------
# Configuración y conexión
# --------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "predictivo_fallas"
COLLECTION_NAME = "sensores"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_fallas.pkl")
WINDOW_SIZE = 50  # número de puntos mostrados en las gráficas

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coleccion = db[COLLECTION_NAME]

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------
# Estructuras compartidas
# --------------------------
lock = Lock()
temperaturas = deque(maxlen=WINDOW_SIZE)
vibraciones = deque(maxlen=WINDOW_SIZE)
voltajes = deque(maxlen=WINDOW_SIZE)
fallas_pred = deque(maxlen=WINDOW_SIZE)
timestamps = deque(maxlen=WINDOW_SIZE)

# Flags y eventos para controlar hilos
sim_running = Event()
pred_running = Event()

# --------------------------
# Utilidades de formato y mensajes
# --------------------------
def actualizar_texto(mensaje):
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    text_area.insert(tk.END, f"[{timestamp}] {mensaje}\n")
    text_area.see(tk.END)

def insertar_mensaje_con_unidades(registro):
    temp = registro["temperatura"]
    vib = registro["vibracion"]
    volt = registro["voltaje"]
    ts = registro.get("timestamp", pd.Timestamp.now())
    mensaje = f"📥 Insertado: T={temp:.2f} °C, Vib={vib:.3f} m/s², Volt={volt:.1f} V @ {ts}"
    actualizar_texto(mensaje)

# --------------------------
# Función: Simulador de datos
# --------------------------
def simulador_datos():
    actualizar_texto("🔁 Simulador iniciado")
    sim_running.set()
    while sim_running.is_set():
        registro = {
            "temperatura": float(np.random.normal(70, 5)),
            "vibracion": float(np.random.normal(0.5, 0.1)),
            "voltaje": float(np.random.normal(220, 10)),
            "timestamp": pd.Timestamp.now()
        }
        coleccion.insert_one(registro)
        insertar_mensaje_con_unidades(registro)
        time.sleep(5)
    actualizar_texto("⏹️ Simulador detenido")

# --------------------------
# Función: Entrenar modelo
# --------------------------
def entrenar_modelo():
    actualizar_texto("⚙️ Iniciando entrenamiento...")
    datos = pd.DataFrame(list(coleccion.find()))
    if datos.empty or len(datos) < 10:
        actualizar_texto("⚠️ No hay datos suficientes para entrenar (mínimo 10 registros).")
        return
    X = datos[["temperatura", "vibracion", "voltaje"]]

    # Si hay etiquetas 'falla', entrenar modelo supervisado; si no, entrenar detector de anomalías
    if "falla" in datos.columns and datos["falla"].notna().sum() >= 10:
        y = datos["falla"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        reporte = classification_report(y_test, y_pred, zero_division=0)

        joblib.dump(modelo, MODEL_PATH)
        actualizar_texto("✅ Modelo supervisado (RandomForest) entrenado y guardado en " + MODEL_PATH)
        for linea in reporte.splitlines():
            actualizar_texto("📊 " + linea)
    else:
        # Entrenamiento no supervisado: detectar anomalías (falla = anomalía)
        from sklearn.ensemble import IsolationForest
        modelo = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        modelo.fit(X)

        joblib.dump(modelo, MODEL_PATH)
        actualizar_texto("✅ Modelo no supervisado (IsolationForest) entrenado y guardado en " + MODEL_PATH)
        actualizar_texto("ℹ️ Entrenamiento sin etiquetas: el modelo detectará anomalías que se interpretan como fallas (1 = anomalía).")

# --------------------------
# Función: Predicción en tiempo real
# --------------------------
def predicciones():
    if not os.path.exists(MODEL_PATH):
        actualizar_texto("⚠️ No se encontró modelo. Entrena el modelo primero.")
        return

    try:
        modelo = joblib.load(MODEL_PATH)
    except Exception as e:
        actualizar_texto(f"❌ Error cargando modelo: {e}")
        return

    actualizar_texto("🔎 Predicción en tiempo real iniciada")
    pred_running.set()
    while pred_running.is_set():
        ultimo = coleccion.find().sort([("_id", -1)]).limit(1)
        for registro in ultimo:
            X_nuevo = pd.DataFrame([{
                "temperatura": registro["temperatura"],
                "vibracion": registro["vibracion"],
                "voltaje": registro["voltaje"]
            }])
            try:
                raw = modelo.predict(X_nuevo)[0]
                model_name = modelo.__class__.__name__
                # Para IsolationForest: devuelve -1 (anomalía) o 1 (normal)
                if model_name == "IsolationForest":
                    pred = 1 if raw == -1 else 0
                else:
                    # Modelos supervisados como RandomForestClassifier devuelven 0/1
                    try:
                        pred = int(raw)
                        pred = 1 if pred == 1 else 0
                    except Exception:
                        pred = 0
            except Exception as e:
                actualizar_texto(f"❌ Error en predicción: {e}")
                pred = 0

            with lock:
                temperaturas.append(registro["temperatura"])
                vibraciones.append(registro["vibracion"])
                voltajes.append(registro["voltaje"])
                fallas_pred.append(pred)
                timestamps.append(str(registro["timestamp"]))

            if pred == 1:
                actualizar_texto(f"⚠️ ALERTA: posible falla en {registro['timestamp']}")
            else:
                actualizar_texto(f"✅ Normal en {registro['timestamp']}")

            # actualizar gráficas desde el hilo principal mediante canvas draw_idle en refresco periódico
        time.sleep(5)
    actualizar_texto("⏹️ Predicción detenida")

# --------------------------
# Función: Exportar datos a CSV
# --------------------------
def exportar_csv():
    datos = pd.DataFrame(list(coleccion.find()))
    if datos.empty:
        actualizar_texto("⚠️ No hay datos para exportar.")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
    if filepath:
        datos.to_csv(filepath, index=False)
        actualizar_texto(f"💾 Datos exportados a {filepath}")

# --------------------------
# Función: Limpiar gráficas y buffers
# --------------------------
def limpiar_datos():
    with lock:
        temperaturas.clear()
        vibraciones.clear()
        voltajes.clear()
        fallas_pred.clear()
        timestamps.clear()
    actualizar_texto("🧹 Gráficas y buffers limpiados")

# --------------------------
# Interfaz gráfica
# --------------------------
root = tk.Tk()
root.title("Monitoreo Predictivo de Fallas")
root.geometry("1150x800")

# Panel superior con botones
frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

btn_start_sim = tk.Button(frame_top, text="Iniciar Simulación", width=18)
btn_stop_sim = tk.Button(frame_top, text="Detener Simulación", width=18)
btn_train = tk.Button(frame_top, text="Entrenar Modelo", width=18)
btn_start_pred = tk.Button(frame_top, text="Iniciar Predicción", width=18)
btn_stop_pred = tk.Button(frame_top, text="Detener Predicción", width=18)
btn_export = tk.Button(frame_top, text="Exportar CSV", width=14)
btn_clear = tk.Button(frame_top, text="Limpiar Gráficas", width=14)

btn_start_sim.grid(row=0, column=0, padx=6, pady=4)
btn_stop_sim.grid(row=0, column=1, padx=6, pady=4)
btn_train.grid(row=0, column=2, padx=6, pady=4)
btn_start_pred.grid(row=0, column=3, padx=6, pady=4)
btn_stop_pred.grid(row=0, column=4, padx=6, pady=4)
btn_export.grid(row=0, column=5, padx=6, pady=4)
btn_clear.grid(row=0, column=6, padx=6, pady=4)

# Área de texto para logs
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=140, height=10)
text_area.pack(padx=10, pady=6, fill=tk.BOTH)

# Área de gráficas
fig, axs = plt.subplots(2, 2, figsize=(10,7))
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt. subplots_adjust(left=0.1)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(padx=10, pady=6, fill=tk.BOTH, expand=True)

# Inicializar títulos con unidades
axs[0,0].set_title("Temperatura (°C)")
axs[0,1].set_title("Vibración (m/s²)")
axs[1,0].set_title("Voltaje (V)")
axs[1,1].set_title("Predicción de Falla (0=Normal, 1=Falla)")

# Función para actualizar las gráficas desde el hilo principal (UI)
def actualizar_grafica():
    with lock:
        t = list(temperaturas)
        v = list(vibraciones)
        vl = list(voltajes)
        f = list(fallas_pred)

    axs[0,0].cla()
    axs[0,0].plot(t, color="red", marker='o', markersize=4)
    axs[0,0].set_title("Temperatura (°C)")
    axs[0,0].set_ylabel("°C")
    axs[0,0].grid(True)

    axs[0,1].cla()
    axs[0,1].plot(v, color="blue", marker='o', markersize=4)
    axs[0,1].set_title("Vibración (m/s²)")
    axs[0,1].set_ylabel("m/s²")
    axs[0,1].grid(True)

    axs[1,0].cla()
    axs[1,0].plot(vl, color="green", marker='o', markersize=4)
    axs[1,0].set_title("Voltaje (V)")
    axs[1,0].set_ylabel("V")
    axs[1,0].grid(True)

    axs[1,1].cla()
    axs[1,1].plot(f, color="purple", marker='o', markersize=6)
    axs[1,1].set_title("Predicción de Falla (0=Normal, 1=Falla)")
    axs[1,1].set_ylim(-0.1, 1.1)
    axs[1,1].set_ylabel("Estado")
    axs[1,1].grid(True)

    canvas.draw_idle()

# Actualización periódica de la gráfica (por si llegan datos desde otros procesos)
def refrescar_periodico():
    actualizar_grafica()
    root.after(2000, refrescar_periodico)  # refrescar cada 2s

# --------------------------
# Conexiones de botones
# --------------------------
def start_sim():
    if sim_running.is_set():
        actualizar_texto("⚠️ Simulador ya está en ejecución.")
        return
    Thread(target=simulador_datos, daemon=True).start()

def stop_sim():
    if sim_running.is_set():
        sim_running.clear()
    else:
        actualizar_texto("⚠️ El simulador no está en ejecución.")

def start_pred():
    if pred_running.is_set():
        actualizar_texto("⚠️ Predicción ya en ejecución.")
        return
    Thread(target=predicciones, daemon=True).start()

def stop_pred():
    if pred_running.is_set():
        pred_running.clear()
    else:
        actualizar_texto("⚠️ La predicción no está en ejecución.")

def train_model():
    Thread(target=entrenar_modelo, daemon=True).start()

def export_csv():
    Thread(target=exportar_csv, daemon=True).start()

def clear_graphs():
    limpiar_datos()
    actualizar_grafica()

btn_start_sim.config(command=start_sim)
btn_stop_sim.config(command=stop_sim)
btn_train.config(command=train_model)
btn_start_pred.config(command=start_pred)
btn_stop_pred.config(command=stop_pred)
btn_export.config(command=export_csv)
btn_clear.config(command=clear_graphs)

# Iniciar refresco periódico y loop principal
root.after(1000, refrescar_periodico)
actualizar_texto("🟢 Interfaz lista. Usa los botones para controlar el sistema.")
root.mainloop()

