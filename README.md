# NFL Win Percentage Prediction - Streamlit App

This repository contains an end-to-end NFL team win percentage analysis and deployment package.

## Contents
- `msis_522_assignment_1.py`: original analysis workflow.
- `nfl_dashboard_pipeline.py`: data prep, model training/evaluation, SHAP generation, artifact saving.
- `train_pipeline.py`: script to pretrain models and generate artifacts.
- `app.py`: Streamlit dashboard (executive summary, descriptive analytics, model performance, explainability, interactive prediction).
- `artifacts/`: pretrained models, metrics, SHAP outputs, and metadata.
- `requirements.txt`: pinned dependencies for reproducibility.

## Dataset
Place these files in the project root:
- `yearly_team_stats_offense.csv`
- `yearly_team_stats_defense.csv`

## Run Locally
1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Pretrain and save artifacts:
   ```bash
   python train_pipeline.py
   ```
3. Launch app:
   ```bash
   python -m streamlit run app.py --server.port 8512
   ```

## Deployment Notes
- Models are pretrained and loaded from `artifacts/`.
- The app should be deployed using `app.py` as the main file.
- `runtime.txt` pins Python version for Streamlit Cloud.
