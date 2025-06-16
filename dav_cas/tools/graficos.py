import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def show_cantidad_clase(y):
    plt.rcParams["figure.figsize"] = (10, 5)
    # summarize distribution
    clases, counts = np.unique(y, return_counts=True)
    clases = [int(clase) for clase in clases]
    porcentaje = [f"{cant/len(y) * 100:.3f} %" for cant in counts]
    data = {"Clase": clases,
            "Cantidad": counts,
            "Porcentaje": porcentaje
            }
    display(pd.DataFrame(data))

    # plot the distribution
    ts = clases
    ts = [str(c) for c in ts]
    plt.bar(ts, counts)
    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def conf_matrix_normalized_sn(y_test, y_pred, title=None):
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Normalizar por filas (clases reales)
    conf_matrix_normalized = conf_matrix.astype(
        'float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, cmap="Blues", fmt=".2f")
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    if title:
        plt.title(title)
    plt.show()
