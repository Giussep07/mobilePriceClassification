from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import os
import pandas as pd
from django.conf import settings


def data_dictionary():
    data_dict = {
        "battery_power": "Energía total que una batería puede almacenar en un momento, medida en mAh",
        "blue": "Indica si tiene Bluetooth o no",
        "clock_speed": "Velocidad a la que el microprocesador ejecuta instrucciones",
        "dual_sim": "Indica si tiene soporte para doble SIM o no",
        "fc": "Megapíxeles de la cámara frontal",
        "four_g": "Indica si tiene 4G o no",
        "int_memory": "Memoria interna en Gigabytes",
        "m_dep": "Profundidad del móvil en cm",
        "mobile_wt": "Peso del teléfono móvil",
        "n_cores": "Número de núcleos del procesador",
        "pc": "Megapíxeles de la cámara principal",
        "px_height": "Resolución de píxeles en altura",
        "px_width": "Resolución de píxeles en anchura",
        "ram": "Memoria de acceso aleatorio en Megabytes",
        "sc_h": "Altura de la pantalla del móvil en cm",
        "sc_w": "Ancho de la pantalla del móvil en cm",
        "talk_time": "Tiempo más largo que durará una sola carga de batería durante una llamada",
        "three_g": "Indica si tiene 3G o no",
        "touch_screen": "Indica si tiene pantalla táctil o no",
        "wifi": "Indica si tiene WiFi o no",
        "price_range": "Esta es la variable objetivo con valores de 0 (bajo costo), 1 (costo medio), 2 (costo alto) y 3 (muy alto costo)"
    }
    return data_dict


@login_required
def index_home(request):
    # ruta al archivo csv
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'train_mobile_price.csv')

    # Leer el archivo csv
    df = pd.read_csv(csv_path)

    # convertir el DataFrame a un diccionario
    data = df.values.tolist()
    columns = df.columns.tolist()

    # Pasar los datos al contexto
    context = {
        'data': data,
        'columns': columns,
        'data_dict': data_dictionary()
    }

    return render(request, template_name="home/index.html", context=context)
