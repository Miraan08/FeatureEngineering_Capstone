import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print accuracy, ROC-AUC, F1 and plot confusion matrix."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"=== {model_name} ===")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
    }


def plot_distribution_comparison(raw_col, scaled_dict, col_name):
    """Plot before/after scaling histograms for a single column."""
    fig, axes = plt.subplots(1, len(scaled_dict) + 1, figsize=(5 * (len(scaled_dict) + 1), 4))
    axes[0].hist(raw_col, bins=40, color="steelblue")
    axes[0].set_title(f"Original — {col_name}")
    for i, (name, vals) in enumerate(scaled_dict.items()):
        axes[i + 1].hist(vals, bins=40, color="darkorange")
        axes[i + 1].set_title(f"{name}\n{col_name}")
    plt.tight_layout()
    plt.show()


def build_group_feature_safe(train_df, test_df, group_col, value_col, agg="mean"):
    """
    Compute group aggregation on TRAIN only, then map to test.
    Prevents data leakage from group-based feature construction.
    """
    group_stats = train_df.groupby(group_col)[value_col].agg(agg)
    train_feature = train_df[group_col].map(group_stats)
    test_feature  = test_df[group_col].map(group_stats)
    return train_feature, test_feature


def construct_features(df):
    """
    Apply all domain-informed feature construction steps.
    Returns the dataframe with new columns added.
    """
    df = df.copy()

    for col in ["adults", "children", "babies", "adr",
                "stays_in_weekend_nights", "stays_in_week_nights",
                "total_of_special_requests", "lead_time", "booking_changes"]:
        if col in df.columns and df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())

    # Ratio features
    df["price_per_person"] = df["adr"] / (
        df["adults"] + df["children"] + df["babies"] + 1
    )
    df["special_requests_rate"] = df["total_of_special_requests"] / (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"] + 1
    )

    # Interaction features
    df["adr_x_lead_time"]    = df["adr"] * df["lead_time"]
    df["changes_x_requests"] = df["booking_changes"] * df["total_of_special_requests"]

    # Derived features
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["is_family"]    = ((df["children"] + df["babies"]) > 0).astype(int)

    return df
