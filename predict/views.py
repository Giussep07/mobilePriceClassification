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


@login_required
def index_predict(request):
    if request.method == "POST":
        # ruta al archivo csv
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'train_mobile_price.csv')

        # Leer el archivo csv
        df = pd.read_csv(csv_path)

        # Seleccionar las características y la variable objetivo
        features = ['ram', 'battery_power', 'px_height', 'px_width', 'mobile_wt']
        x = df[features]
        y = df['price_range']

        # 80% para entrenamiento y 20% para prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

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

        searchResults = grid_search.fit(x_train_boost, y_train.to_numpy().ravel())

        # extract the best model and evaluate it
        bestModel = searchResults.best_estimator_

        # Se crea un objeto con los mejores ajustes de Hiperparámetros
        dtc = bestModel

        # Se entrena el modelo con los mejores parámetros
        dtc.fit(x_train_boost, y_train)

        # Se genera la prediccion con los datos del form
        ram = int(request.POST['ram'])
        battery_power = int(request.POST['battery_power'])
        px_height = int(request.POST['px_height'])
        px_width = int(request.POST['px_width'])
        mobile_wt = int(request.POST['mobile_wt'])
        test_data = [ram, battery_power, px_height, px_width, mobile_wt]
        new_features = pd.DataFrame([test_data], columns=features)

        new_features_scaled = scaler.transform(new_features)

        # Hacer la predicción
        new_prediction = bestModel.predict(new_features_scaled)

        if new_prediction[0] == 0:
            category = "Bajo costo"
        elif new_prediction[0] == 1:
            category = "Costo medio"
        elif new_prediction[0] == 2:
            category = "Costo alto"
        else:
            category = "Muy alto costo"

        print("Antes del context")
        context = {
            'prediction': new_prediction[0],
            'category': category
        }

        print("Antes del render, context: ", context)
        return render(request, template_name="predict/index.html", context=context)

    return render(request, template_name="predict/index.html")
