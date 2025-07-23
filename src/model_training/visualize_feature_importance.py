# visualize_feature_importance.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# defined directories
MODEL_DIR = "/Users/anuragchaubey/RouteWise/models"
PLOT_DIR = "/Users/anuragchaubey/RouteWise/outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

def visualize_feature_importance(
    model_names: List[str],
    feature_names: pd.Index,
    top_n: int = 10
) -> Dict[str, plt.Figure]:
    """
    Generates and saves top-N feature importance barplots for compatible models.
    """
    if not isinstance(feature_names, pd.Index):
        raise TypeError("feature_names must be a pandas Index.")

    plots = {}
    print("\n generating feature importance plots...")

    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if not os.path.isfile(model_path):
            print(f" Skipping {model_name}: Model not found at {model_path}")
            continue

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f" Error loading {model_name}: {e}")
            continue

        if not hasattr(model, 'feature_importances_'):
            print(f" {model_name} does not support feature_importances_. Skipping.")
            continue

        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            print(f"Mismatch in feature count for {model_name}. Skipping.")
            continue

        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=df_importance,hue='feature', palette='viridis',legend=False, ax=ax)
        ax.set_title(f'Top {top_n} Features - {model_name}')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, f"{model_name}_top_{top_n}_features.png")
        plt.savefig(plot_path)
        plt.close(fig)
        plots[model_name] = fig
        print(f" {model_name} plot saved: {plot_path}")

    return plots
