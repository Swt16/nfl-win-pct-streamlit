from nfl_dashboard_pipeline import load_and_merge_data, save_artifacts, train_and_evaluate


def main() -> None:
    data = load_and_merge_data("yearly_team_stats_offense.csv", "yearly_team_stats_defense.csv")
    results = train_and_evaluate(data)
    meta = save_artifacts("artifacts", data, results, report_text_path="Dataset_Introduction.txt")

    print("Training complete.\n")
    print(results["metrics"].to_string(index=False))
    print("\nBest model:", meta["best_model"])
    print("SHAP status:", meta["shap"])


if __name__ == "__main__":
    main()
