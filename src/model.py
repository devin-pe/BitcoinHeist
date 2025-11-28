import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import inv_link
from configs.configs import RunConfig
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve
)

def train_and_explain(train_X, train_y):
    model = ExplainableBoostingClassifier(
      random_state=RunConfig.seed, 
      n_jobs=-1,
      max_rounds=100,  
      early_stopping_rounds=50 
    )
    model.fit(train_X, train_y)
    
    global_explanation = model.explain_global(name='EBM')
    global_explanation_data = global_explanation.data()
    global_importance_dict = dict(zip(global_explanation_data['names'], 
                                      global_explanation_data['scores']))
    
    local_explanation_values = model.eval_terms(train_X.values)
    local_attributions = pd.DataFrame(
        local_explanation_values,
        columns=model.term_names_,
        index=train_X.index
    )
    
    raw_preds = local_explanation_values.sum(axis=1) + model.intercept_
    probabilities = inv_link(raw_preds, model.link_)[:, 1] # only prob for ransomware
    
    return model, probabilities, global_importance_dict, local_attributions
  

def get_optimal_threshold(y_true, probabilities):
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Threshold: {optimal_threshold}")
    
    return optimal_threshold
  
  
def evaluate(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probabilities)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "probabilities": y_pred,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    