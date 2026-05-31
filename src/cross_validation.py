from dataclasses import dataclass, field

import pandas as pd 
import numpy as np

from typing import Any

from sklearn.model_selection import KFold
from .utils import Cols

import logging
logger = logging.getLogger(__name__)

from sklearn.base import clone

from sklearn.metrics import (
    # classification 
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    # regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)

#### Dataclasses ####

@dataclass
class ModelConfig:
    model:Any
    regression:bool
    features:list|None = None 
    preprocessing:Any|None = None 
    threshold_classification:float=0.5
    threshold_regression:float=0


@dataclass
class CVconfig:
    n_splits: int = 8
    shuffle: bool = True
    random_state: int = 0
    verbose: bool = False

@dataclass
class CVresults:
    n_folds:int 
    oof_results: pd.DataFrame 
    fold_results:list
    mean_accuracy: float = np.nan

    def get_fold_results(self, i:int):
        return self.fold_results[i]
    
    def score(self, penalty: float = 1.0) -> float:
        acc_by_date = self.oof_results.groupby("TS")["is_correct"].mean()
        mean = acc_by_date.mean()
        std = acc_by_date.std()
        score = mean - penalty * std
        print(f"mean={mean:.4f}  std={std:.4f}  score(penalty={penalty})={score:.4f}")
        return score

@dataclass
class FoldResults:
    fold_id:int
    train_metrics: dict
    valid_metrics: dict
    train_index: pd.Index 
    valid_index: pd.Index 
    train_dates: list 
    valid_dates: list 
    fitted_model: Any = None        
    y_score_train: pd.Series = None 
    y_score_test: pd.Series = None  
    y_binary_train: pd.Series = None
    y_binary_test: pd.Series = None


#### Main function ####

def run_cv(X, y, model_config, cv_config=None):

    if cv_config is None:
        cv_config = CVconfig()  

    if cv_config.verbose:
        logger.setLevel(logging.INFO)

    _check_data(X, y)

    date_col = Cols.DATE.value
    unique_ts = X[date_col].unique()

    splitter = KFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state,
    )

    oof_results = pd.DataFrame(
        {
            "fold": np.nan, 
            "TS": None,
            "ALLOCATION": None,
            "y_true": np.nan, 
            "y_true_binarized": np.nan, 
            "score": np.nan, 
            "prediction": np.nan,
            "is_correct": np.nan,
        }, 
        index=y.index
    )

    fold_results = []

    tgt = Cols.TARGET.value
    tgt_bin = Cols.TARGET_BIN.value

    for fold_id, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_ts), start=1):

        logger.info(f"Processing Fold {fold_id}/{cv_config.n_splits}")

        train_dates = unique_ts[train_date_idx]
        valid_dates = unique_ts[valid_date_idx]
        train_mask = X[date_col].isin(train_dates)
        valid_mask = X[date_col].isin(valid_dates)
        X_train_fold = X.loc[train_mask]
        X_test_fold = X.loc[valid_mask]
        y_train_fold = y.loc[train_mask]
        y_test_fold = y.loc[valid_mask]

        X_train_ready, X_test_ready = _preprocess_fold_data(X_train_fold, X_test_fold, model_config=model_config) 
        local_model, y_score_train, y_score_test = _fit_model_and_predict(X_train_ready, X_test_ready, y_train_fold, model_config=model_config) 

        if model_config.regression:
            y_binary_train = (y_score_train > model_config.threshold_regression).astype(int)
            y_binary_test = (y_score_test > model_config.threshold_regression).astype(int)
            train_metrics = _score_model_regression(y_train_fold[tgt], y_score_train) | _score_model_classification(y_train_fold[tgt_bin], y_binary_train)
            valid_metrics = _score_model_regression(y_test_fold[tgt], y_score_test) | _score_model_classification(y_test_fold[tgt_bin], y_binary_test)
        else: 
            y_binary_train = (y_score_train > model_config.threshold_classification).astype(int)
            y_binary_test = (y_score_test > model_config.threshold_classification).astype(int)
            train_metrics = _score_model_classification(y_train_fold[tgt_bin], y_binary_train, y_score_train)
            valid_metrics = _score_model_classification(y_test_fold[tgt_bin], y_binary_test, y_score_test)

        fold_result = FoldResults(
            fold_id=fold_id,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            train_index=train_mask,
            valid_index=valid_mask,
            train_dates=train_dates,
            valid_dates=valid_dates,
            fitted_model=local_model,
            y_score_train=y_score_train, 
            y_score_test=y_score_test, 
            y_binary_train=y_binary_train, 
            y_binary_test=y_binary_test,

        )

        fold_results.append(fold_result)

        oof_results.loc[valid_mask, "fold"] = fold_id
        oof_results.loc[valid_mask, "TS"] = X_test_fold[Cols.DATE.value].values
        oof_results.loc[valid_mask, "ALLOCATION"] = X_test_fold[Cols.ALLOCATION.value].values
        oof_results.loc[valid_mask, "y_true"] = y_test_fold[tgt]
        oof_results.loc[valid_mask, "y_true_binarized"] = y_test_fold[tgt_bin]
        oof_results.loc[valid_mask, "score"] = y_score_test
        oof_results.loc[valid_mask, "prediction"] = y_binary_test
        
        logger.info(f"Fold accuracy={valid_metrics['accuracy']:.2f}")

    oof_results["is_correct"] = (oof_results["prediction"] == oof_results["y_true_binarized"]).astype(int)

    output = CVresults(
        n_folds=cv_config.n_splits,
        oof_results=oof_results,
        fold_results=fold_results,
        mean_accuracy=oof_results["is_correct"].mean()
    )

    return output 

#### helpers ####

def _check_data(X, y):
    assert X.index.equals(y.index), "X and y do not share the same index"
    assert (Cols.TARGET.value in y.columns) and (Cols.TARGET_BIN.value in y.columns), f"y columns are incorrect {y.columns}"
    assert Cols.DATE.value in X.columns, "X has no date column"
    
def _preprocess_fold_data(X_train_fold, X_test_fold, model_config):
    
    # Sélection features d'abord pour éviter de passer des strings au preprocessing
    if model_config.features is not None:
        X_train_fold = X_train_fold[model_config.features]
        X_test_fold = X_test_fold[model_config.features]
    else:
        drop_cols = [c for c in [Cols.ALLOCATION.value, Cols.DATE.value, Cols.GROUP.value] if c in X_train_fold.columns]
        X_train_fold = X_train_fold.drop(columns=drop_cols)
        X_test_fold = X_test_fold.drop(columns=drop_cols)

    if model_config.preprocessing is not None:
        X_train_fold, X_test_fold = model_config.preprocessing(X_train_fold, X_test_fold)

    return X_train_fold, X_test_fold

def _fit_model_and_predict(X_train_ready, X_test_ready, y_train_ready, model_config):
    local_model = clone(model_config.model)

    if model_config.regression:
        local_model.fit(X_train_ready, y_train_ready[Cols.TARGET.value])
        y_score_train = local_model.predict(X_train_ready)
        y_score_test = local_model.predict(X_test_ready)
        return local_model, y_score_train, y_score_test

    else: 
        local_model.fit(X_train_ready, y_train_ready[Cols.TARGET_BIN.value])
        y_score_train = local_model.predict_proba(X_train_ready)[:, 1]
        y_score_test = local_model.predict_proba(X_test_ready)[:, 1]
        return local_model, y_score_train, y_score_test


def _score_model_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def _score_model_classification(y_true, y_pred_binary, y_pred_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "confusion_matrix":confusion_matrix(y_true, y_pred_binary)
    }
    if y_pred_score is not None:
        metrics["auc"] = roc_auc_score(y_true, y_pred_score)
        metrics["brier"] = brier_score_loss(y_true, y_pred_score)
        metrics["log_loss"] = log_loss(y_true, y_pred_score)
    return metrics


    
    
    