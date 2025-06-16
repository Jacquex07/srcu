import pandas as pd  # módulo para tratamiento de datos
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler


def cod_one_hot_ecoder(df, lst_car_cat):
    lst_categories = []
    for cat in lst_car_cat:
        lst_categories.append(list(df[cat].unique()))

    enc = OneHotEncoder(categories=lst_categories)
    datos_cod_ohe = enc.fit_transform(df[lst_car_cat])
    labels_cod_ohe = enc.get_feature_names_out(lst_car_cat)
    df_cod_ohe = pd.DataFrame(datos_cod_ohe.toarray(),
                              columns=labels_cod_ohe,
                              index=df.index)
    return df_cod_ohe, enc


def cod_one_hot_ecoder_2(df, lst_car_cat, ohe):
    res_ohe = ohe.transform(df[lst_car_cat])
    labels_cod_ohe = ohe.get_feature_names_out(lst_car_cat)
    df_encoded = pd.DataFrame(data=res_ohe.toarray(), columns=labels_cod_ohe)
    return df_encoded


def cod_ordinal_ecoder(df, dic_car_ord):
    enc = OrdinalEncoder(categories=list(dic_car_ord.values()))
    datos_cod_ord = enc.fit_transform(df[list(dic_car_ord.keys())])
    labels_cod_ord = enc.get_feature_names_out(list(dic_car_ord.keys()))
    df_cod_ord = pd.DataFrame(datos_cod_ord,
                              columns=labels_cod_ord,
                              index=df.index)
    return df_cod_ord, enc


def cod_ordinal_ecoder_2(df, lst_car_ord, enc):
    res_enc = enc.transform(df[lst_car_ord])
    labels_cod_enc = enc.get_feature_names_out(lst_car_ord)
    df_encoded = pd.DataFrame(data=res_enc, columns=labels_cod_enc)
    return df_encoded


def cod_label_ecoder(df, nom_clase):
    enc = LabelEncoder()
    datos_cod_lab = enc.fit_transform(df[nom_clase])
    df_cod_lab = pd.DataFrame(datos_cod_lab,
                              columns=[nom_clase],
                              index=df.index)
    return df_cod_lab, enc


def cod_label_ecoder_2(df, nom_clase, enc):
    res_enc = enc.transform(df[nom_clase])
    df_encoded = pd.DataFrame(data=res_enc, columns=[nom_clase])
    return df_encoded


def esc_robust_scaler(df, lst_car_num):
    norm_rob_sca = RobustScaler()
    df_rob_sca = pd.DataFrame(norm_rob_sca.fit_transform(df[lst_car_num]),
                              columns=norm_rob_sca.feature_names_in_,
                              index=df.index)
    return df_rob_sca, norm_rob_sca


def esc_robust_scaler_2(df, lst_car_num, res):
    esc_res = res.transform(df[lst_car_num])
    labels_esc_rob = res.feature_names_in_
    df_scaled = pd.DataFrame(data=esc_res, columns=labels_esc_rob)
    return df_scaled


def unificacion_codificados(lista_dfs):
    if len(lista_dfs) > 1:
        df_res = lista_dfs[0]
        for i in range(1, len(lista_dfs)):
            df_res = df_res.join(lista_dfs[i])
        return df_res


def get_programa_cod(mod_cod_oen_clase, value):
    try:
        return mod_cod_oen_clase.categories_[0][value]
    except IndexError:
        return f'{value}: Codificación Inválida. Valores soportados en el rango de 0 a 38'


def get_programa_cod_lab(mod_cod_len_clase, value):
    try:
        return mod_cod_len_clase.inverse_transform([value])[0]
    except IndexError:
        return f'{value}: Codificación Inválida. Valores soportados en el rango de 0 a 38'