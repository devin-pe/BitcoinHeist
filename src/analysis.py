import pandas as pd
import numpy as np

def determine_important_features(y_true: pd.Series, probabilities: np.ndarray, 
                                 local_attributions: pd.DataFrame, threshold: float
                                ) -> dict[str, dict[str, float]]:
    predictions = (probabilities >= threshold)
    tp_mask = (y_true == 1) & (predictions == 1)
    fn_mask = (y_true == 1) & (predictions == 0)

    feature_info = {}

    tp_attributions = local_attributions.loc[tp_mask]
    mean_tp_impact = tp_attributions.mean(axis=0).sort_values(ascending=True).head(5)
    feature_info["tp_weakest_5"] = mean_tp_impact.to_dict()


    fn_attributions = local_attributions.loc[fn_mask]
    mean_fn_impact = fn_attributions.mean(axis=0).sort_values(ascending=True).head(5)
    feature_info["fn_strongest_5"] = mean_fn_impact.to_dict()

    return feature_info