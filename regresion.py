# ==========================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================

import pandas as pd                  # Manejo de datos en estructuras tipo DataFrame
import numpy as np                   # Operaciones numéricas
import matplotlib.pyplot as plt      # Gráficas
import seaborn as sns                # Gráficas estadísticas más estilizadas
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.linear_model import LinearRegression     # Modelo de regresión lineal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas

import tkinter as tk                 # Para crear la interfaz gráfica
from tkinter import messagebox       # Para mostrar mensajes emergentes


# ==========================
# FUNCIÓN PARA CARGAR DATOS
# ==========================

def load_data(dataset):
    """
    Carga el archivo CSV dependiendo de la opción seleccionada.
    """
    if dataset == 'home':
        data = pd.read_csv('home-rental.csv')
    elif dataset == 'ice':
        data = pd.read_csv('ice-cream.csv')
    else:
        raise ValueError("Opción inválida.")
    
    return data


# ==========================
# FUNCIÓN PARA REALIZAR REGRESIÓN
# ==========================

def perform_regression(data, target_column=None, feature_columns=None):
    """
    Realiza una regresión lineal sobre el dataset recibido.
    """

    # Selecciona únicamente columnas numéricas
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("El dataset no contiene columnas numéricas.")

    # Si no se especifica variable objetivo, toma la última numérica
    if target_column is None:
        target_column = numeric_cols[-1]

    # Las variables independientes serán todas menos la variable objetivo
    if feature_columns is None:
        feature_columns = [c for c in numeric_cols if c != target_column]

    # X = variables independientes
    X = data[feature_columns]

    # y = variable dependiente (lo que queremos predecir)
    y = data[target_column]

    # Divide los datos:
    # 80% entrenamiento
    # 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crear modelo de regresión lineal
    model = LinearRegression()

    # Entrenar modelo con datos de entrenamiento
    model.fit(X_train, y_train)

    # Hacer predicciones con los datos de prueba
    predictions = model.predict(X_test)

    # Calcular métricas de evaluación
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return predictions, y_test, mae, rmse, r2


# ==========================
# FUNCIÓN PARA VISUALIZAR RESULTADOS
# ==========================

def visualize_results(y_test, predictions):
    """
    Muestra una gráfica comparando valores reales vs predichos.
    """

    plt.figure(figsize=(10, 6))

    # regplot dibuja puntos y línea de tendencia
    sns.regplot(
        x=y_test,
        y=predictions,
        scatter_kws={'color': 'blue'},
        line_kws={'color': 'red'}
    )

    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Resultados de la Regresión')
    plt.show()


# ==========================
# FUNCIÓN PRINCIPAL
# ==========================

def run_model(choice):
    """
    Ejecuta todo el proceso cuando el usuario selecciona un dataset.
    """

    try:
        # Cargar datos
        data = load_data(choice)

        # Mostrar columnas disponibles
        print("\nColumnas del dataset:", list(data.columns))

        numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        print("Columnas numéricas:", numeric)

        # Usar última columna numérica como variable objetivo
        target = numeric[-1]
        print(f"Usando '{target}' como variable objetivo.")

        # Ejecutar regresión
        predictions, y_test, mae, rmse, r2 = perform_regression(data, target_column=target)

        # Mostrar métricas
        print(f"\nMean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R² score: {r2}\n")

        # Mostrar gráfica
        visualize_results(y_test, predictions)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ==========================
# INTERFAZ GRÁFICA
# ==========================

# Crear ventana principal
root = tk.Tk()
root.title("Modelo de Regresión Lineal")
root.geometry("300x200")

# Etiqueta
label = tk.Label(root, text="Selecciona el Dataset", font=("Arial", 12))
label.pack(pady=10)

# Botón para dataset Home Rental
btn_home = tk.Button(
    root,
    text="Home Rental",
    command=lambda: run_model('home')
)
btn_home.pack(pady=5)

# Botón para dataset Ice Cream
btn_ice = tk.Button(
    root,
    text="Ice Cream",
    command=lambda: run_model('ice')
)
btn_ice.pack(pady=5)

# Ejecutar interfaz
root.mainloop()