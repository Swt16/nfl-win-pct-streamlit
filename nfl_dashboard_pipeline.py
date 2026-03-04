import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


KEYS = ["team", "season"]
TARGET = "win_pct"
DEFAULT_FEATURES = [
    "total_off_points",
    "total_tds",
    "td_pct",
    "pass_pct_defense",
    "rec_td_pct",
    "receiving_touchdown",
    "total_off_yards",
    "ypa",
]


@dataclass
class PreparedData:
    merged: pd.DataFrame
    model_df: pd.DataFrame
    feature_candidates: List[str]


def sanitize_name(name: str) -> str:
    safe = name.lower().replace(" ", "_").replace("/", "_")
    safe = safe.replace("(", "").replace(")", "")
    safe = re.sub(r"[^a-z0-9_]+", "_", safe)
    return safe.strip("_")


def load_and_merge_data(offense_csv: str, defense_csv: str) -> PreparedData:
    off = pd.read_csv(offense_csv)
    deff = pd.read_csv(defense_csv)

    off = off.loc[off["season_type"] == "REG"].copy()
    deff = deff.loc[deff["season_type"] == "REG"].copy()

    merged = off.merge(deff, on=KEYS, how="left", suffixes=("_offense", "_defense"))

    if "win_pct_offense" in merged.columns:
        merged = merged.rename(columns={"win_pct_offense": "win_pct"})
    if "win_pct_defense" in merged.columns:
        merged = merged.drop(columns=["win_pct_defense"])

    if TARGET not in merged.columns:
        raise ValueError("Target column 'win_pct' not found after merge.")

    feature_candidates = [
        c for c in merged.columns
        if c not in {"team", "season", TARGET, "season_type_offense", "season_type_defense"}
    ]

    model_cols = [c for c in DEFAULT_FEATURES if c in merged.columns] + [TARGET, "team", "season"]
    model_df = merged[model_cols].dropna(subset=[TARGET]).copy()

    return PreparedData(merged=merged, model_df=model_df, feature_candidates=feature_candidates)


def _fit_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, {}

    if name == "Tuned Lasso Regression":
        base = Lasso(random_state=42, max_iter=2000)
        grid = GridSearchCV(
            base,
            {"alpha": np.logspace(-4, 0, 20)},
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    if name == "Tuned Ridge Regression":
        base = Ridge(random_state=42, max_iter=2000)
        grid = GridSearchCV(
            base,
            {"alpha": np.logspace(-4, 2, 20)},
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    if name == "Tuned Elastic-Net Regression":
        base = ElasticNet(random_state=42, max_iter=2000)
        grid = GridSearchCV(
            base,
            {
                "alpha": np.logspace(-4, 0, 10),
                "l1_ratio": np.arange(0.1, 1.0, 0.2),
            },
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    if name == "Decision Tree (CART)":
        base = DecisionTreeRegressor(random_state=42)
        grid = GridSearchCV(
            base,
            {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [5, 10, 20, 50],
            },
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    if name == "Random Forest":
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = GridSearchCV(
            base,
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 8, None],
            },
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_

    if name == "Boosted Trees":
        if LGBMRegressor is not None:
            base = LGBMRegressor(random_state=42, verbose=-1)
            grid = GridSearchCV(
                base,
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1],
                },
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            return grid.best_estimator_, grid.best_params_

        if XGBRegressor is not None:
            base = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
            grid = GridSearchCV(
                base,
                {
                    "n_estimators": [100, 300],
                    "max_depth": [3, 4, 5],
                    "learning_rate": [0.03, 0.1],
                },
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            return grid.best_estimator_, grid.best_params_

        raise RuntimeError("Neither LightGBM nor XGBoost is installed.")

    if name == "Neural Network":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            max_iter=100,
            batch_size=32,
            early_stopping=False,
            random_state=42,
        )
        mlp.fit(Xs, y_train)
        return {"model": mlp, "scaler": scaler}, {}

    raise ValueError(f"Unsupported model: {name}")


def train_and_evaluate(data: PreparedData, selected_features: List[str] | None = None) -> Dict:
    features = selected_features if selected_features else [c for c in DEFAULT_FEATURES if c in data.model_df.columns]

    work = data.model_df[features + [TARGET]].copy()
    X = work[features]
    y = work[TARGET].astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.3, random_state=42)

    model_names = [
        "Linear Regression",
        "Tuned Lasso Regression",
        "Tuned Ridge Regression",
        "Tuned Elastic-Net Regression",
        "Decision Tree (CART)",
        "Random Forest",
        "Boosted Trees",
        "Neural Network",
    ]

    trained = {}
    rows = []

    for name in model_names:
        model_obj, best_params = _fit_model(name, X_train, y_train)

        if name == "Neural Network":
            preds = model_obj["model"].predict(model_obj["scaler"].transform(X_test))
        else:
            preds = model_obj.predict(X_test)

        rows.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "r2": float(r2_score(y_test, preds)),
                "best_params": json.dumps(best_params),
            }
        )
        trained[name] = model_obj

    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    return {
        "features": features,
        "imputer": imputer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "models": trained,
    }


def _save_shap_for_model(model_name: str, model_obj, X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: Path):
    import shap

    X_sample = X_test.sample(min(120, len(X_test)), random_state=42)

    if model_name == "Neural Network" and isinstance(model_obj, dict):
        bg = X_train.sample(min(100, len(X_train)), random_state=42).copy()
        pred_fn = lambda data: model_obj["model"].predict(
            model_obj["scaler"].transform(pd.DataFrame(data, columns=X_train.columns))
        )
        explainer = shap.Explainer(pred_fn, bg)
        shap_values = explainer(X_sample)
        values = shap_values.values
        plot_data = X_sample
    else:
        if isinstance(model_obj, dict):
            return {"status": "skipped", "reason": "Unsupported wrapped model for SHAP."}
        explainer = shap.Explainer(model_obj, X_train)
        shap_values = explainer(X_sample)
        values = shap_values.values
        plot_data = X_sample

    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

    safe = sanitize_name(model_name)
    csv_path = out_dir / f"shap_importance_{safe}.csv"
    png_path = out_dir / f"shap_summary_{safe}.png"

    importance.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, plot_data, feature_names=X_sample.columns.tolist(), show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return {"status": "ok", "csv": csv_path.name, "png": png_path.name}


def extract_report_text(text_path: str) -> str:
    p = Path(text_path)
    if not p.exists():
        return "Dataset_Introduction.txt not found."

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")


def save_artifacts(
    out_dir: str,
    data: PreparedData,
    results: Dict,
    report_text_path: str = "Dataset_Introduction.txt",
) -> Dict:
    out = Path(out_dir)
    model_dir = out / "models"
    shap_dir = out / "shap"
    model_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)

    data.merged.to_csv(out / "merged_regular_season.csv", index=False)
    data.model_df.to_csv(out / "modeling_view.csv", index=False)
    results["metrics"].to_csv(out / "metrics.csv", index=False)

    joblib.dump(results["imputer"], out / "imputer.joblib")

    model_map = {}
    for model_name, model_obj in results["models"].items():
        safe = sanitize_name(model_name)
        fpath = model_dir / f"{safe}.joblib"
        joblib.dump(model_obj, fpath)
        model_map[model_name] = fpath.name

    shap_status = {}
    try:
        best_tree = None
        for candidate in results["metrics"]["model"].tolist():
            if candidate in {"Decision Tree (CART)", "Random Forest", "Boosted Trees"}:
                best_tree = candidate
                break

        for m in ["Linear Regression", best_tree, "Neural Network"]:
            if m is None:
                continue
            shap_status[m] = _save_shap_for_model(m, results["models"][m], results["X_train"], results["X_test"], shap_dir)
    except Exception as exc:
        shap_status["error"] = str(exc)

    report_text = extract_report_text(report_text_path)
    (out / "executive_summary.txt").write_text(report_text, encoding="utf-8")

    meta = {
        "rows_regular_season": int(len(data.merged)),
        "modeling_rows": int(len(data.model_df)),
        "all_feature_candidates": data.feature_candidates,
        "selected_features": results["features"],
        "model_files": model_map,
        "best_model": results["metrics"].iloc[0]["model"],
        "shap": shap_status,
    }

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
