from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings

# Libreria para cálculos numéricos
import numpy as np
# Libreria para manipulación y análisis de datos
import pandas as pd
from home.views import data_dictionary

# Libreria para el preprocesamiento de los datos
from sklearn import preprocessing
# Libreria para el balanceo de los datos
from sklearn.utils import resample
# Librería para separar los datos de entrenamiento y de pruebas
from sklearn.model_selection import train_test_split
# Libreria para el modelo de clasificación Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Libreria para la matriz de confusión
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Libreria para el reporte de clasificación
from sklearn.metrics import classification_report
# Librería para las métricas: Precision, Recall y F1 Score
from sklearn.metrics import precision_score, recall_score, f1_score

# Libreria árbol de decisión para selección de mejores características y predicción
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Libreria para calcular la media y la desviación estándar utilizadas en las características
from sklearn.preprocessing import StandardScaler
# Libreria de búsqueda en cuadrícula
from sklearn.model_selection import GridSearchCV


@login_required
def index_nb(request):
    # ruta al archivo csv
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'train_mobile_price.csv')

    # Leer el archivo csv
    df = pd.read_csv(csv_path)

    # convertir el DataFrame a un diccionario
    data = df.values.tolist()
    columns = df.columns.tolist()

    # Seleccionar las características y la variable objetivo
    features = ['ram', 'battery_power', 'px_height', 'px_width', 'mobile_wt']
    x = df[features]
    y = df['price_range']

    # 80% para entrenamiento y 20% para prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Se crea el clasificador de tipo Naive Bayes
    nb = GaussianNB()

    # Se entrena el modelo
    nb.fit(x_train, y_train)

    # Se genera la predicción
    prediction = nb.predict(x_test)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test, prediction)

    # Crear una lista de filas de la matriz de confusión con etiquetas
    conf_matrix_with_labels = []
    for row, label in zip(conf_matrix, nb.classes_):
        conf_matrix_with_labels.append((label, row.tolist()))

    # Se imprimen las métricas
    precision = round(precision_score(y_test, prediction, average='macro') * 100, 2)
    recall = round(recall_score(y_test, prediction, average='macro') * 100, 2)
    score = round(f1_score(y_test, prediction, average='macro') * 100, 2)

    # Pesos de las características
    weights = {
        "ram": round(0.6192 * 100, 2),
        "battery_power": round(0.1319 * 100, 2),
        "px_height": round(0.0894 * 100, 2),
        "px_width": round(0.0793 * 100, 2),
        "mobile_wt": round(0.0201 * 100, 2),
        "clock_speed": round(0.0128 * 100, 2),
        "int_memory": round(0.0083 * 100, 2),
        "sc_h": round(0.0070 * 100, 2),
        "fc": round(0.0065 * 100, 2),
        "m_dep": round(0.0062 * 100, 2),
        "talk_time": round(0.0053 * 100, 2),
        "dual_sim": round(0.0039 * 100, 2),
        "sc_w": round(0.0028 * 100, 2),
        "pc": round(0.0028 * 100, 2),
        "four_g": round(0.0028 * 100, 2),
        "n_cores": round(0.0015 * 100, 2),
        "three_g": round(0.0004 * 100, 2),
        "touch_screen": 0,
        "blue": 0,
        "wifi": 0
    }

    # Hiperparámetros con GridSearch
    scaler = StandardScaler()
    x_train_boost = scaler.fit_transform(x_train)
    x_test_boost = scaler.transform(x_test)

    # Se crea el clasificador de tipo Naive Bayes
    nb_boost = GaussianNB()

    # Parámetros
    params_nb = {'var_smoothing': np.logspace(0, -9, num=100)}

    grid_search = GridSearchCV(estimator=nb_boost,
                               param_grid=params_nb,
                               cv=10,
                               verbose=1,
                               n_jobs=-1,
                               scoring="accuracy")

    searchResults = grid_search.fit(x_train_boost, y_train.ravel())

    # extract the best model and evaluate it
    bestModel = searchResults.best_estimator_

    # Se crea un objeto con los mejores ajustes de Hiperparámetros
    dtc = bestModel

    # Se entrena el modelo con los mejores parámetros
    dtc.fit(x_train_boost, y_train)

    pred = dtc.predict(x_test_boost)

    # Calcular la matriz de confusión
    conf_matrix_boost = confusion_matrix(y_test, pred)

    # Crear una lista de filas de la matriz de confusión con etiquetas
    conf_matrix_with_labels_boost = []
    for row, label in zip(conf_matrix_boost, nb.classes_):
        conf_matrix_with_labels_boost.append((label, row.tolist()))

    # Otras métricas clasificación: Precisión, Recall, F1-Score
    # Se genera la precisión
    precision_boost = round(precision_score(y_test, pred, average='macro') * 100, 2)

    # Se genera la sensibilidad
    recall_boost = round(recall_score(y_test, pred, average='macro') * 100, 2)

    # Se genera la puntuación F1
    score_boost = round(f1_score(y_test, pred, average='macro') * 100, 2)

    context = {
        'data': data,
        'columns': columns,
        'data_dict': data_dictionary(),
        'weights': weights,
        'confusion_matrix_with_labels': conf_matrix_with_labels,
        'labels': nb.classes_,
        'precision': precision,
        'recall': recall,
        'score': score,
        'confusion_matrix_with_labels_boost': conf_matrix_with_labels_boost,
        'precision_boost': precision_boost,
        'recall_boost': recall_boost,
        'score_boost': score_boost
    }

    return render(request, template_name="nb/index.html", context=context)
