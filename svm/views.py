from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings

## Libreria para operaciones matematicas o estadisticas
import numpy as np
## Libreria para el manejo de datos
import pandas as pd

## Libreria para cambios de datos (LabelEncoder)
from sklearn.preprocessing import LabelEncoder
## Libreria para balanceo de datos
from sklearn.utils import resample
## Libreria arbol de decision para seleccion de mejores caracteristicas
from sklearn.tree import DecisionTreeClassifier, plot_tree
## Libreria para separar los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
## Libreria para SVM
from sklearn.svm import SVC
## Libreria para metricas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Libreria para calcular la media y la desviación estándar utilizadas en las características
from sklearn.preprocessing import StandardScaler
# Libreria de búsqueda en cuadrícula
from sklearn.model_selection import GridSearchCV

from home.views import data_dictionary


@login_required
def index_svm(request):
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

    # Se define el algoritmo SVC
    svm = SVC()

    # Se entrena el modelo
    svm.fit(x_train, y_train)

    # Se genera la prediccion
    predict = svm.predict(x_test)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test, predict)

    # Crear una lista de filas de la matriz de confusión con etiquetas
    conf_matrix_with_labels = []
    for row, label in zip(conf_matrix, svm.classes_):
        conf_matrix_with_labels.append((label, row.tolist()))

    # Calcular el classification report
    report = classification_report(y_test, predict)

    # Se genera la precisión
    precision = round(precision_score(y_test, predict, average='macro') * 100, 2)

    # Se genera la sensibilidad
    recall = round(recall_score(y_test, predict, average='macro') * 100, 2)

    # Se genera la puntuación F1
    score = round(f1_score(y_test, predict, average='macro') * 100, 2)

    # Pesos de las características
    weights = {
        'ram': round(0.6215 * 100, 2),
        'battery_power': round(0.1383 * 100, 2),
        'px_height': round(0.0948 * 100, 2),
        'px_width': round(0.0776 * 100, 2),
        'mobile_wt': round(0.0173 * 100, 2),
        'int_memory': round(0.0094 * 100, 2),
        'talk_time': round(0.0074 * 100, 2),
        'clock_speed': round(0.0062 * 100, 2),
        'sc_h': round(0.0053 * 100, 2),
        'm_dep': round(0.0052 * 100, 2),
        'fc': round(0.0041 * 100, 2),
        'dual_sim': round(0.0039 * 100, 2),
        'sc_w': round(0.0030 * 100, 2),
        'touch_screen': round(0.0025 * 100, 2),
        'pc': round(0.0016 * 100, 2),
        'n_cores': round(0.0007 * 100, 2),
        'wifi': round(0.0006 * 100, 2),
        'blue': round(0.0006 * 100, 2),
        'four_g': 0,
        'three_g': 0,
    }

    # Hiperparámetros con GridSearch
    scaler = StandardScaler()
    x_train_boost = scaler.fit_transform(x_train)
    x_test_boost = scaler.transform(x_test)

    svm_boost = SVC()

    # Parámetros
    params_SVC = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(estimator=svm_boost,
                               param_grid=params_SVC,
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
    for row, label in zip(conf_matrix_boost, svm.classes_):
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
        'labels': svm.classes_,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'score': score,
        'confusion_matrix_with_labels_boost': conf_matrix_with_labels_boost,
        'precision_boost': precision_boost,
        'recall_boost': recall_boost,
        'score_boost': score_boost
    }

    return render(request, template_name="svm/index.html", context=context)
