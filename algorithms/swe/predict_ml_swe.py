# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from algorithms.swe.snowai.swe.machine_learning_model import MachineLearningSWE
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================
# 参数
# =========================
IN_CSV = os.path.join(BASE_DIR, "output", "ml_inputs.csv")
OUT_CSV = os.path.join(BASE_DIR, "output", "swe_ml_distribution.csv")


def clean_for_ml(df: pd.DataFrame) -> pd.DataFrame:

    required = [
        "Snow_Class",
        "Elevation_m",
        "Snow_Depth_m",
        "TAVG_degC",
        "TMIN_degC",
        "TMAX_degC",
        "datetime",
    ]

    df = df.copy()

    # 时间
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    # Snow class
    df["Snow_Class"] = df.get("Snow_Class", "alpine")
    df["Snow_Class"] = df["Snow_Class"].fillna("alpine").astype(str)

    # 数值列
    for c in ["Elevation_m", "Snow_Depth_m", "TAVG_degC", "TMIN_degC", "TMAX_degC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除 NaN
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


def main():

    df = pd.read_csv(IN_CSV)

    df = clean_for_ml(df)

    # 预测 ML SWE
    preds = df.assign(
        SWE_ML_cm=lambda x: MachineLearningSWE(return_type="pandas").predict(
            data=x,
            snow_class="Snow_Class",
            elevation="Elevation_m",
            snow_depth="Snow_Depth_m",
            tavg="TAVG_degC",
            tmin="TMIN_degC",
            tmax="TMAX_degC",
            DOY="datetime",
        )
    )

    # 如果存在 ERA5 SWE，则保留
    if "SWE_ERA5_cm" in preds.columns:
        cols = [
            "datetime",
            "Longitude",
            "Latitude",
            "Elevation_m",
            "Snow_Depth_m",
            "SWE_ERA5_cm",
            "SWE_ML_cm",
            "TAVG_degC",
            "TMIN_degC",
            "TMAX_degC",
        ]
        cols = [c for c in cols if c in preds.columns]
        preds = preds[cols]

    preds.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("[INFO] Saved:", OUT_CSV)
    print(preds.head())


if __name__ == "__main__":
    main()