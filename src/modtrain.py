from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import numpy as np

def train_and_evaluate(X, y):
    """Versión optimizada sin warnings"""
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        tree_method='hist',  # Más eficiente
        random_state=42
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        preds = model.predict(X_val)
        qwk_scores.append(cohen_kappa_score(y_val, preds, weights='quadratic'))
    
    return model, np.mean(qwk_scores)