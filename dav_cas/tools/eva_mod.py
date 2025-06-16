import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter


def show_evaluaciones(punt_eval):
    tam = len(punt_eval['Precision ma'])
    d_mod = range(1, tam + 1)
    display(pd.DataFrame(punt_eval, d_mod))


def eval_f1_mcc_ck(dy_test, dy_pred):
    punt_eval = {"F1-score ma": [],
                 "Matthews corr": [],
                 "Cohen kappa": []
                 }
    num_modelos = len(dy_test)
    for idat in range(1, num_modelos + 1):
        punt_eval['F1-score ma'].append(f1_score(dy_test[idat],
                                        dy_pred[idat], average='macro'))
        punt_eval['Matthews corr'].append(
            matthews_corrcoef(dy_test[idat], dy_pred[idat]))
        punt_eval['Cohen kappa'].append(
            cohen_kappa_score(dy_test[idat], dy_pred[idat]))

    tam = len(punt_eval['F1-score ma'])
    d_mod = range(1, tam + 1)
    display(pd.DataFrame(punt_eval, d_mod).apply(lambda x: round(x, 4)))


def eval_modelos(dy_test, dy_pred):
    punt_eval = {"Precision ma": [],
                 "Precision we": [],
                 "Precision mi": [],
                 "Recall ma": [],
                 "Recall we": [],
                 "Recall mi": [],
                 "F1-score ma": [],
                 "F1-score we": [],
                 "F1-score mi": []
                 }
    num_modelos = len(dy_test)
    for idat in range(1, num_modelos + 1):
        # Métricas de evaluación con 'macro' para multi-clase con clases de igual importancia
        # Baseline precision score
        punt_eval['Precision ma'].append(precision_score(
            dy_test[idat], dy_pred[idat], average="macro"))
        punt_eval['Precision we'].append(precision_score(
            dy_test[idat], dy_pred[idat], average="weighted"))
        punt_eval['Precision mi'].append(precision_score(
            dy_test[idat], dy_pred[idat], average="micro"))

        # Baseline recall score
        punt_eval['Recall ma'].append(recall_score(
            dy_test[idat], dy_pred[idat], average="macro"))
        punt_eval['Recall we'].append(recall_score(
            dy_test[idat], dy_pred[idat], average="weighted"))
        punt_eval['Recall mi'].append(recall_score(
            dy_test[idat], dy_pred[idat], average="micro"))

        # Baseline f1 score
        punt_eval['F1-score ma'].append(f1_score(dy_test[idat],
                                        dy_pred[idat], average="macro"))
        punt_eval['F1-score we'].append(f1_score(dy_test[idat],
                                        dy_pred[idat], average="weighted"))
        punt_eval['F1-score mi'].append(f1_score(dy_test[idat],
                                        dy_pred[idat], average="micro"))
    return punt_eval


def show_mejor_dataset(punt_eval):
    lst_datos_score = []
    print(f"{'Evaluación':<20}{'Datos':<9} {'Score':<10}")
    print(f"{'-'*13:<20}{'-'*5:<9} {'-'*6:<10}")
    best_score_f1_ma = 0
    for score in punt_eval.keys():
        val_max = max(punt_eval[score])
        best_dataset = punt_eval[score].index(val_max) + 1
        if score == "F1-score ma":
            best_score_f1_ma = best_dataset
        print(f"{score:<15}{best_dataset:>10} {val_max:>10.4f}")
        lst_datos_score.append(f"{punt_eval[score].index(val_max) + 1}")
    return best_score_f1_ma, lst_datos_score


def frec_dataset_eval(lst_datos_score):
    c = Counter(lst_datos_score)
    print("Tabla de Frecuencia Datos en Evaluaciones")
    display(pd.DataFrame({"Datos": c.keys(), "Frecuencia": c.values()}))


def conf_matrix__class_repo(idat, dy_test, dy_pred):
    print(confusion_matrix(dy_test[idat], dy_pred[idat]))
    print(classification_report(dy_test[idat], dy_pred[idat]))


def pipeline_x_train_x_test_tranformed(pipeline, X_train, X_test, name_preprocessor='preprocessor'):
    # Extraer el ColumnTransformer del pipeline
    preprocessor = pipeline.named_steps[name_preprocessor]
    # Transformar los datos de entrenamiento
    X_train_transformed = preprocessor.transform(X_train)
    # Transformar los datos de prueba
    X_test_transformed = preprocessor.transform(X_test)
    # Obtener nombres de las características transformadas
    feature_names = preprocessor.get_feature_names_out()
    # Crear DataFrames
    df_train_transformed = pd.DataFrame(
        X_train_transformed, columns=feature_names)
    df_test_transformed = pd.DataFrame(
        X_test_transformed, columns=feature_names)
    display("Datos de entrenamiento transoformados (X_train_transformed):")
    display(df_train_transformed)
    display("Datos de prueba transoformados (X_test_transformed):")
    display(df_test_transformed)


def classification_report_df(y_test, y_pred, precisión=2):
    # Genera el reporte como diccionario
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Extrae la exactitud
    accuracy = report_dict['accuracy']

    # Convierte a DataFrame y transpón
    df = pd.DataFrame(report_dict).transpose()

    for eval in ['precision', 'recall', 'f1-score']:
        df[eval] = df[eval].apply(lambda x: f'{x:,.{precisión}f}')
        df[eval] = df[eval].astype(str)
    df['support'] = df['support'].astype(int)

    # Agrega la fila de exactitud
    df.loc["accuracy"] = [
        '', '', f'{accuracy:,.{precisión}f}', df.loc["macro avg", "support"]]

    # Reinicia el índice y renombra
    df.reset_index(inplace=True)
    df.rename(columns={"index": "class"}, inplace=True)

    # Muestra el DataFrame
    display(df.apply(lambda x: x))
