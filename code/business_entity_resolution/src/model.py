import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.dummy import DummyClassifier
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def f05_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame, y_val: pd.Series) -> lgb.Booster:
    print("Training LightGBM model...")
    
    if y_train.sum() == 0:
        print("No positive samples, using DummyClassifier")
        model = DummyClassifier(strategy='constant', constant=0)
        model.fit(X_train, y_train)
        return model
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4,
        'seed': 42,
        'scale_pos_weight': len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks
    )
    
    val_pred = (model.predict(X_val, num_iteration=model.best_iteration) > 0.5).astype(int)
    print(f"Validation Precision: {precision_score(y_val, val_pred):.4f}")
    print(f"Validation Recall: {recall_score(y_val, val_pred):.4f}")
    print(f"Validation F0.5: {f05_score(y_val, val_pred):.4f}")
    
    return model


def find_optimal_threshold(model, X_val: pd.DataFrame, 
                            y_val: pd.Series) -> float:
    print("Finding optimal threshold for F0.5...")
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, 'predict'):
        probs = model.predict(X_val)
        if probs.ndim > 1:
            probs = probs[:, 1]
    else:
        probs = model.predict(X_val, num_iteration=model.best_iteration)
    
    best_thresh = 0.5
    best_f05 = 0.0
    
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (probs > thresh).astype(int)
        f05 = f05_score(y_val, preds)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
    
    print(f"Optimal threshold: {best_thresh:.3f}, F0.5: {best_f05:.4f}")
    return best_thresh


def predict_candidates(model, X: pd.DataFrame, 
                        pairs: List[Tuple[str, str]], 
                        threshold: float) -> Dict[str, List[str]]:
    print("Running inference...")
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        if probs.shape[1] > 1:
            probs = probs[:, 1]
        else:
            probs = probs[:, 0]
    elif hasattr(model, 'best_iteration'):
        probs = model.predict(X, num_iteration=model.best_iteration)
    else:
        probs = model.predict(X)
        if probs.ndim > 1:
            probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    preds = (probs > threshold).astype(int)
    
    matches = defaultdict(list)
    for (s1_id, cand_id), pred in zip(pairs, preds):
        if pred == 1:
            matches[s1_id].append(cand_id)
    
    return dict(matches)


def save_model(model, threshold: float, path: str):
    model_data = {
        'model': model,
        'threshold': threshold
    }
    joblib.dump(model_data, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Tuple[object, float]:
    model_data = joblib.load(path)
    return model_data['model'], model_data['threshold']


from collections import defaultdict