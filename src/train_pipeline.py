import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def run_pipeline():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "time-series-19-covid-combined.csv")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    df_nacional = df.groupby(["Country/Region", "Date"]).agg({
        "Confirmed": "sum", "Recovered": "sum", "Deaths": "sum", "Lat": "first", "Long": "first"
    }).reset_index().sort_values(["Country/Region", "Date"]).reset_index(drop=True)

    df_nacional["Recovered"] = df_nacional.groupby("Country/Region")["Recovered"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both").fillna(0)
    )
    df_nacional["Daily_Confirmed"] = df_nacional.groupby("Country/Region")["Confirmed"].diff().fillna(0).clip(lower=0)
    df_nacional["Daily_Recovered"] = df_nacional.groupby("Country/Region")["Recovered"].diff().fillna(0).clip(lower=0)
    df_nacional["Daily_Deaths"] = df_nacional.groupby("Country/Region")["Deaths"].diff().fillna(0).clip(lower=0)
    df_nacional["Fatality_Rate"] = np.where(df_nacional["Confirmed"] > 0, (df_nacional["Deaths"] / df_nacional["Confirmed"]) * 100, 0.0)
    df_nacional["Confirmed_MA7"] = df_nacional.groupby("Country/Region")["Daily_Confirmed"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_nacional["Active_Cases_Est"] = (df_nacional["Confirmed"] - df_nacional["Recovered"] - df_nacional["Deaths"]).clip(lower=0)
    ratio = np.where(df_nacional["Confirmed_MA7"] > 0, df_nacional["Daily_Confirmed"] / df_nacional["Confirmed_MA7"], 1.0)
    df_nacional["Daily_Growth_Ratio"] = np.clip(ratio, 0.0, 10.0)

    umbral = df_nacional["Daily_Deaths"].quantile(0.75)
    df_nacional["Risk_Level"] = (df_nacional["Daily_Deaths"] > umbral).astype(int)

    features = ["Confirmed", "Recovered", "Daily_Confirmed", "Daily_Recovered", 
                "Fatality_Rate", "Confirmed_MA7", "Active_Cases_Est", "Daily_Growth_Ratio", "Lat", "Long"]

    X = df_nacional[features].copy()
    y_reg = df_nacional["Daily_Deaths"].values
    y_clf = df_nacional["Risk_Level"].values

    X_train, _, y_train_reg, _, y_train_clf, _ = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_scaled, y_train_reg)

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train_clf)

    dash_dir = os.path.join(base_path, "dashboard")
    joblib.dump(rf_reg, os.path.join(dash_dir, "modelo_regresion_rf.pkl"))
    joblib.dump(rf_clf, os.path.join(dash_dir, "modelo_clasificacion_rf.pkl"))
    joblib.dump(scaler, os.path.join(dash_dir, "escalador_robusto.pkl"))
    df_nacional.to_csv(os.path.join(base_path, "data", "datos_epidemiologicos_limpios.csv"), index=False)
    print("Pipeline ejecutado y artefactos actualizados.")

if __name__ == "__main__":
    run_pipeline()
