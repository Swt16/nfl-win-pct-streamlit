import html
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

from nfl_dashboard_pipeline import load_and_merge_data, train_and_evaluate

ART = Path("artifacts")
MODELS = ART / "models"
FIGSIZE = (6, 3.4)
ENABLE_CUSTOM_MLP_TRAINER = False
REPORT_COMMENTS = {
    "target_distribution": (
        "The distribution of win percentages looks fairly normal shaped; median is centered "
        "around 0.5, with no major skew, outliers, or imbalance."
    ),
    "points_vs_win": (
        "As expected, the more points a team scores, the better the win percentage; from an "
        "eyeball view, around 375 points is near a 0.5 win percentage."
    ),
    "season_box": (
        "The median win percentage stays near 0.5 by season, but variability shifts year to year, "
        "with more spread in recent seasons."
    ),
    "top_bottom_teams": (
        "Top-five and bottom-five team win percentages are close to mirror images around the 0.5 line."
    ),
    "points_by_season": (
        "Average offensive points trend upward over time; this aligns with the 17th game era and "
        "recent NFL scoring environment."
    ),
    "corr_heatmap": (
        "Offensive metrics strongly correlate with each other and with win percentage; defensive "
        "coverage in this dataset is more limited."
    ),
    "pairplot": (
        "Key metrics move positively with win percentage and appear broadly well-behaved in distribution."
    ),
    "model_table": (
        "Tuned Lasso performed best overall in the report, with other linear models close behind; "
        "ensemble methods improved over single-tree performance."
    ),
    "rmse_bar": (
        "Model RMSE comparison reflects that simpler linear approaches were competitive with, or "
        "better than, higher-complexity models on this feature set."
    ),
    "diag_generic": (
        "Predicted-vs-actual alignment shows model fit quality; tighter clustering around the diagonal "
        "indicates better calibration."
    ),
    "tree_plot": (
        "A constrained tree is interpretable but can underfit compared with stronger ensemble or linear baselines."
    ),
    "shap_bar": (
        "SHAP feature-importance results indicate total_off_points and pass_pct_defense as the strongest drivers."
    ),
    "shap_beeswarm": (
        "The SHAP summary pattern is consistent with the bar ranking: the top two features dominate contribution."
    ),
    "shap_waterfall": (
        "The waterfall view decomposes one prediction into baseline plus feature-level pushes up or down."
    ),
}

st.set_page_config(page_title="NFL Win% Dashboard", layout="wide")
st.set_option("global.dataFrameSerialization", "legacy")


def read_text_with_fallback(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


@st.cache_data
def load_artifact_tables():
    metrics = pd.read_csv(ART / "metrics.csv")
    modeling_df = pd.read_csv(ART / "modeling_view.csv")
    merged = pd.read_csv(ART / "merged_regular_season.csv")
    with open(ART / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    intro_path = Path("Dataset_Introduction.txt")
    if intro_path.exists():
        summary_text = read_text_with_fallback(intro_path)
    else:
        summary_text = read_text_with_fallback(ART / "executive_summary.txt")
    return metrics, modeling_df, merged, meta, summary_text


def _norm_title(text: str) -> str:
    t = (text or "").lower().strip()
    t = t.replace("rsme", "rmse")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_data
def load_plot_comments() -> dict:
    p = Path("Plot_Comments.csv")
    if not p.exists():
        return {}
    df = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        return {}
    if "Plot_Title" not in df.columns or "Comment" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[_norm_title(str(row["Plot_Title"]))] = str(row["Comment"]).strip()
    return out


def show_comment(plot_comments: dict, title: str, fallback: str = "") -> None:
    comment = plot_comments.get(_norm_title(title), fallback)
    if comment:
        normalized_lines = []
        for line in str(comment).splitlines():
            line = line.replace("\u00a0", " ").replace("\t", " ")
            line = re.sub(r"^\s+", "", line)
            normalized_lines.append(line)
        while normalized_lines and normalized_lines[0] == "":
            normalized_lines.pop(0)
        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()

        collapsed = []
        prev_blank = False
        for line in normalized_lines:
            is_blank = (line == "")
            if is_blank and prev_blank:
                continue
            collapsed.append(line)
            prev_blank = is_blank

        rendered = []
        for i, line in enumerate(collapsed):
            if i > 0:
                rendered.append("<br>")
            if line:
                rendered.append(html.escape(line))
        safe_comment = "".join(rendered)
        st.markdown("**Commentary:**")
        st.markdown(
            f'<div style="line-height: 1.45; margin-bottom: 0.75rem;">{safe_comment}</div>',
            unsafe_allow_html=True,
        )


@st.cache_resource
def load_models(meta: dict):
    model_objs = {}
    for name, file_name in meta.get("model_files", {}).items():
        path = MODELS / file_name
        if path.exists():
            model_objs[name] = joblib.load(path)
    imputer = joblib.load(ART / "imputer.joblib")
    return model_objs, imputer


def model_predictions_for_plot(model_obj, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model_obj, dict):
        return model_obj["model"].predict(model_obj["scaler"].transform(X))
    return model_obj.predict(X)


def train_custom_keras_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    units_1: int,
    units_2: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(units_1, activation="relu"),
            tf.keras.layers.Dense(units_2, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    history = model.fit(
        X_train_scaled,
        y_train.values,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    preds = model.predict(X_test_scaled, verbose=0).flatten()
    results = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }
    return history.history, results


@st.cache_resource
def build_explainer(model_name: str, _model_obj, _X_background: pd.DataFrame):
    import shap

    if isinstance(_model_obj, dict):
        pred_fn = lambda data: _model_obj["model"].predict(_model_obj["scaler"].transform(pd.DataFrame(data, columns=_X_background.columns)))
        return shap.Explainer(pred_fn, _X_background)

    return shap.Explainer(_model_obj, _X_background)


def render_executive_summary(metrics: pd.DataFrame, meta: dict, summary_text: str):
    st.subheader("Executive Summary")

    best = metrics.sort_values("rmse").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best["model"])
    c2.metric("Best RMSE", f"{best['rmse']:.4f}")
    c3.metric("Best R2", f"{best['r2']:.4f}")

    st.caption(
        f"Rows used (REG only): {meta.get('rows_regular_season')} | "
        f"Modeling rows: {meta.get('modeling_rows')}"
    )

    st.markdown(
        """
        <style>
        .exec-summary-wrap {
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 12px 14px;
            background: #fff;
            max-height: 520px;
            overflow-y: auto;
        }
        .exec-summary-body {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.55;
            white-space: pre-wrap;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    lines = str(summary_text).splitlines()
    first_line = lines[0] if lines else ""
    rest = "\n".join(lines[1:]) if len(lines) > 1 else ""
    rendered_text = (
        f"<strong>{html.escape(first_line)}</strong>"
        + (f"<br>{html.escape(rest)}" if rest else "")
    )
    st.markdown(
        f"""
        <div class="exec-summary-wrap">
            <div class="exec-summary-body">{rendered_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_descriptive_analytics(modeling_df: pd.DataFrame, plot_comments: dict):
    st.subheader("Descriptive Analytics")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(modeling_df["win_pct"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Win Percentage")
    ax.set_xlabel("Win Percentage")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Distribution of Win Percentage", REPORT_COMMENTS["target_distribution"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.regplot(x="total_off_points", y="win_pct", data=modeling_df, ax=ax, scatter_kws={"alpha": 0.6})
    ax.set_title("Total Offensive Points vs Win Percentage")
    ax.set_xlabel("Total Offensive Points")
    ax.set_ylabel("Win Percentage")
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Total Offensive Points vs. Win Percentage with Regression Line", REPORT_COMMENTS["points_vs_win"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(x="season", y="win_pct", data=modeling_df, ax=ax)
    ax.set_title("Win Percentage Distribution Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Win Percentage")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Win Percentage Distribution Across Seasons", REPORT_COMMENTS["season_box"])

    avg_win_pct_by_team = modeling_df.groupby("team")["win_pct"].mean().sort_values(ascending=False)
    top_5_teams = avg_win_pct_by_team.head(5)
    bottom_5_teams = avg_win_pct_by_team.tail(5)
    combined_teams = pd.concat([top_5_teams, bottom_5_teams])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x=combined_teams.index, y=combined_teams.values, ax=ax, hue=combined_teams.index, legend=False)
    ax.set_title("Average Win Percentage: Top 5 vs Bottom 5 Teams")
    ax.set_xlabel("Team")
    ax.set_ylabel("Average Win Percentage")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Average Win Percentage: Top 5 vs. Bottom 5 Teams", REPORT_COMMENTS["top_bottom_teams"])

    table_c1, table_c2 = st.columns(2)
    with table_c1:
        st.markdown("**Top 5 Teams by Average Win%**")
        st.dataframe(top_5_teams.reset_index(name="avg_win_pct"), use_container_width=True)
    with table_c2:
        st.markdown("**Bottom 5 Teams by Average Win%**")
        st.dataframe(bottom_5_teams.reset_index(name="avg_win_pct"), use_container_width=True)

    avg_off_points_by_season = modeling_df.groupby("season")["total_off_points"].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(x="season", y="total_off_points", data=avg_off_points_by_season, marker="o", ax=ax)
    ax.set_title("Average Total Offensive Points by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Total Offensive Points")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Average Total Offensive Points by Season", REPORT_COMMENTS["points_by_season"])

    corr_input = modeling_df.drop(columns=["team", "season"], errors="ignore")
    corr = corr_input.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Key Features and Win Percentage")
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Correlation Matrix of Key Features and Win Percentage", REPORT_COMMENTS["corr_heatmap"])

    pair_cols = [c for c in ["total_off_points", "td_pct", "pass_pct_defense", "win_pct"] if c in modeling_df.columns]
    if len(pair_cols) >= 3:
        st.markdown("**Pair Plot (Key Features)**")
        pair_fig = sns.pairplot(modeling_df[pair_cols], height=1.5)
        st.pyplot(pair_fig.fig)
        plt.close(pair_fig.fig)
        show_comment(plot_comments, "Pair Plot of Key Offensive/Defensive Metrics and Win Percentage", REPORT_COMMENTS["pairplot"])


def render_model_performance(metrics: pd.DataFrame, modeling_df: pd.DataFrame, meta: dict, models: dict, imputer, plot_comments: dict):
    st.subheader("Model Performance")

    st.markdown("**Model Comparison Table**")
    st.dataframe(metrics, use_container_width=True)
    st.caption(REPORT_COMMENTS["model_table"])

    rmse_mae_long = metrics.melt(
        id_vars="model",
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=rmse_mae_long, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("RMSE and MAE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rmse_sorted = metrics.sort_values("rmse", ascending=True)
    sns.barplot(data=rmse_sorted, x="rmse", y="model", hue="model", legend=False, ax=ax)
    ax.set_title("Comparison of Model RMSE on Test Set")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Model")
    st.pyplot(fig)
    plt.close(fig)
    show_comment(plot_comments, "Comparison of Model RMSE on Test Set", REPORT_COMMENTS["rmse_bar"])

    st.markdown("**Best Hyperparameters by Model**")
    hyper_table = metrics[["model", "best_params"]].copy()
    st.dataframe(hyper_table, use_container_width=True)

    features = meta.get("selected_features", [])
    if not features:
        return

    X_full = pd.DataFrame(imputer.transform(modeling_df[features]), columns=features)
    y_full = modeling_df["win_pct"].astype(float)
    _, X_test_diag, _, y_test_diag = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

    st.markdown("**Predicted vs Actual Plots (All Models)**")
    diag_order = [
        "Linear Regression",
        "Tuned Lasso Regression",
        "Tuned Ridge Regression",
        "Tuned Elastic-Net Regression",
        "Decision Tree (CART)",
        "Random Forest",
        "Boosted Trees",
        "Neural Network",
    ]
    for diag_model in diag_order:
        if diag_model in models:
            y_pred_diag = model_predictions_for_plot(models[diag_model], X_test_diag)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.scatter(y_test_diag, y_pred_diag, alpha=0.6)
            lim_min = min(float(np.min(y_test_diag)), float(np.min(y_pred_diag)))
            lim_max = max(float(np.max(y_test_diag)), float(np.max(y_pred_diag)))
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2)
            display_name_map = {
                "Tuned Lasso Regression": "Tuned Lasso",
                "Tuned Ridge Regression": "Tuned Ridge",
                "Tuned Elastic-Net Regression": "Tuned Elastic-Net",
            }
            title_name = display_name_map.get(diag_model, diag_model)
            ax.set_title(f"{title_name}: Predicted vs Actual Win Percentage")
            ax.set_xlabel("Actual Win Percentage")
            ax.set_ylabel("Predicted Win Percentage")
            st.pyplot(fig)
            plt.close(fig)
            title_map = {
                "Random Forest": "Random Forest Regressor: Predicted vs Actual Win Percentage",
                "Boosted Trees": "LightGBM Regressor: Predicted vs. Actual Win Percentage",
                "Neural Network": "Neural Network: Predicted vs Actual Win Percentage",
            }
            if diag_model == "Tuned Elastic-Net Regression":
                show_comment(
                    plot_comments,
                    "Comparison of Model RMSE on Test Set",
                    REPORT_COMMENTS["diag_generic"],
                )
            elif diag_model not in {"Tuned Lasso Regression", "Tuned Ridge Regression"}:
                show_comment(plot_comments, title_map.get(diag_model, diag_model), REPORT_COMMENTS["diag_generic"])

    if "Decision Tree (CART)" in models:
        dt_model = models["Decision Tree (CART)"]
        if not isinstance(dt_model, dict):
            st.markdown("**Decision Tree (CART) Structure**")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(
                dt_model,
                feature_names=features,
                filled=True,
                rounded=True,
                fontsize=5,
                max_depth=3,
                ax=ax,
            )
            ax.set_title("Decision Tree Visualization (Depth Capped at 3)")
            st.pyplot(fig)
            plt.close(fig)
            show_comment(plot_comments, "Best Decision Tree Regressor (Max Depth: 3, Min Samples Leaf: 5)", REPORT_COMMENTS["tree_plot"])


def render_explainability_and_interactive(metrics: pd.DataFrame, modeling_df: pd.DataFrame, meta: dict, models: dict, imputer, plot_comments: dict):
    st.subheader("Explainability & Interactive Prediction")

    st.markdown("**Interactive Prediction**")
    model_name = st.selectbox("Select model", options=list(models.keys()))
    features = meta.get("selected_features", [])

    if not features:
        st.warning("No selected features found in metadata.")
        return

    default_adjustable = features[: min(4, len(features))]
    adjustable = st.multiselect(
        "Choose features to manually set",
        options=features,
        default=default_adjustable,
    )

    base_vals = modeling_df[features].mean(numeric_only=True)
    custom_vals = {}
    for feat in features:
        min_v = float(modeling_df[feat].min())
        max_v = float(modeling_df[feat].max())
        default_v = float(base_vals[feat]) if feat in base_vals else min_v
        if feat in adjustable:
            custom_vals[feat] = st.slider(
                feat,
                min_value=min_v,
                max_value=max_v,
                value=float(np.clip(default_v, min_v, max_v)),
            )
        else:
            custom_vals[feat] = default_v

    custom_input_raw = pd.DataFrame([custom_vals], columns=features)
    custom_input = pd.DataFrame(imputer.transform(custom_input_raw), columns=features)

    selected_model = models[model_name]
    prediction = float(model_predictions_for_plot(selected_model, custom_input)[0])
    st.metric("Predicted Win%", f"{prediction:.4f}")

    if ENABLE_CUSTOM_MLP_TRAINER:
        st.markdown("**Custom MLP (Keras) Trainer**")
        tf_available = True
        try:
            import tensorflow as tf  # noqa: F401
        except Exception:
            tf_available = False

        if not tf_available:
            st.info(
                "TensorFlow is not installed. Install it to enable configurable Keras MLP training: "
                "`python -m pip install tensorflow`"
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                units_1 = st.slider("Hidden Layer 1 Units", min_value=32, max_value=256, value=128, step=16)
                units_2 = st.slider("Hidden Layer 2 Units", min_value=32, max_value=256, value=128, step=16)
            with c2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                    value=0.001,
                )
                batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
            with c3:
                epochs = st.slider("Epochs", min_value=20, max_value=300, value=100, step=10)
                validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2, step=0.05)

            if st.button("Train Custom Keras MLP"):
                X_full = pd.DataFrame(imputer.transform(modeling_df[features]), columns=features)
                y_full = modeling_df["win_pct"].astype(float)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=0.3, random_state=42
                )
                history, test_metrics = train_custom_keras_mlp(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    units_1=units_1,
                    units_2=units_2,
                    learning_rate=float(learning_rate),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=float(validation_split),
                )

                h1, h2, h3 = st.columns(3)
                h1.metric("Test MAE", f"{test_metrics['mae']:.4f}")
                h2.metric("Test RMSE", f"{test_metrics['rmse']:.4f}")
                h3.metric("Test R2", f"{test_metrics['r2']:.4f}")

                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(history.get("loss", []), label="Train Loss")
                ax.plot(history.get("val_loss", []), label="Val Loss")
                ax.set_title("Keras MLP Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE Loss")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=FIGSIZE)
                ax.plot(history.get("mae", []), label="Train MAE")
                ax.plot(history.get("val_mae", []), label="Val MAE")
                ax.set_title("Keras MLP MAE Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MAE")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("**SHAP Waterfall for Custom Input**")
    try:
        import shap

        bg = pd.DataFrame(imputer.transform(modeling_df[features]), columns=features).sample(min(120, len(modeling_df)), random_state=42)
        explainer = build_explainer(model_name, selected_model, bg)
        exp = explainer(custom_input)

        plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(exp[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(REPORT_COMMENTS["shap_waterfall"])
    except Exception as exc:
        st.warning(f"Could not render SHAP waterfall for this model/input: {exc}")

    shap_status = meta.get("shap", {})
    st.markdown("**SHAP Summary Outputs**")
    st.json(shap_status)

    shap_dir = ART / "shap"
    files = sorted(shap_dir.glob("shap_importance_*.csv"))
    if not files:
        st.info("No SHAP artifacts found. Re-run training after installing shap.")

    for f in files:
        st.markdown(f"**{f.name}**")
        sdf = pd.read_csv(f)
        st.dataframe(sdf.head(20), use_container_width=True)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        top_bar = sdf.head(15).sort_values("mean_abs_shap", ascending=True)
        sns.barplot(data=top_bar, x="mean_abs_shap", y="feature", ax=ax)
        ax.set_title(f"SHAP Bar Plot ({f.stem.replace('shap_importance_', '')})")
        ax.set_xlabel("Mean Absolute SHAP")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(REPORT_COMMENTS["shap_bar"])

        png = shap_dir / f.name.replace("shap_importance_", "shap_summary_").replace(".csv", ".png")
        if png.exists():
            st.image(str(png), caption=png.name, width=900)
            st.caption(REPORT_COMMENTS["shap_beeswarm"])

    show_comment(plot_comments, "SHAP Analysis", REPORT_COMMENTS["shap_bar"])


def main():
    st.title("NFL Regular Season Win Percentage Modeling")

    if not ART.exists():
        st.error("Artifacts not found. Run `python train_pipeline.py` first.")
        return

    metrics, modeling_df, merged, meta, summary_text = load_artifact_tables()
    models, imputer = load_models(meta)
    plot_comments = load_plot_comments()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ])

    with tab1:
        render_executive_summary(metrics, meta, summary_text)

    with tab2:
        render_descriptive_analytics(modeling_df, plot_comments)

    with tab3:
        render_model_performance(metrics, modeling_df, meta, models, imputer, plot_comments)

    with tab4:
        render_explainability_and_interactive(metrics, modeling_df, meta, models, imputer, plot_comments)


if __name__ == "__main__":
    main()
