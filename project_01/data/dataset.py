# data/dataset.py
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str = "aux_database_.json"):
    with open(path, "r") as f:
        datos_json = json.load(f)

    df = pd.DataFrame(datos_json)
    df_training = df[["task_name", "start_time", "end_time"]].copy()

    df_training["start_time"]    = pd.to_datetime(df_training["start_time"])
    df_training["end_time"]      = pd.to_datetime(df_training["end_time"])
    df_training["duration_mins"] = (
        (df_training["end_time"] - df_training["start_time"]).dt.total_seconds() / 60.0
    )
    df_training["day_of_week"] = df_training["start_time"].dt.dayofweek
    df_training["hour"]        = df_training["start_time"].dt.hour
    df_training["minute"]      = df_training["start_time"].dt.minute

    le = LabelEncoder()
    df_training["task_id"] = le.fit_transform(df_training["task_name"])

    df_training = df_training.sort_values("start_time").reset_index(drop=True)

    es_nueva_semana = df_training["day_of_week"] < df_training["day_of_week"].shift(1)
    es_nueva_semana.iloc[0] = False
    df_training["week_id"] = es_nueva_semana.cumsum()

    semanas_separadas = [
        df_training[df_training["week_id"] == i]
        for i in range(df_training["week_id"].max() + 1)
        if not df_training[df_training["week_id"] == i].empty
    ]

    print(f"Total de semanas extraídas: {len(semanas_separadas)}")
    return df_training, semanas_separadas


df_training, semanas_separadas = load_data()