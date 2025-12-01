import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import inv_link
from configs.configs import RunConfig
import numpy as np
from typing import Union
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    roc_curve
)

def train(train_X: pd.DataFrame, train_y: pd.Series) \
        -> tuple[ExplainableBoostingClassifier, np.ndarray, dict, pd.DataFrame]:
            
    model = ExplainableBoostingClassifier(
      random_state=RunConfig.seed, 
      n_jobs = RunConfig.n_jobs,
      max_rounds= RunConfig.max_rounds,  
    )
    model.fit(train_X, train_y)
    return model


def explain(model: ExplainableBoostingClassifier, train_X: pd.DataFrame) \
                                    -> tuple[np.ndarray, pd.DataFrame]:
    
    local_explanation_values = model.eval_terms(train_X.values)
    local_attributions = pd.DataFrame(
        local_explanation_values,
        columns=model.term_names_,
        index=train_X.index
    )
    
    raw_preds = local_explanation_values.sum(axis=1) + model.intercept_
    probabilities = inv_link(raw_preds, model.link_)[:, 1] # only prob for ransomware
    
    return probabilities, local_attributions
  

def get_optimal_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
  
  
def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) \
        -> dict[str, Union[float, np.ndarray]]:
            
    y_pred = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    _, _, fn, tp = conf_matrix.ravel()
    return {
        "accuracy": accuracy,
        "fn": fn,
        "tp": tp,
    }
    
    