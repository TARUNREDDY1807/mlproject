import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    """Fit each model (optionally with hyperparameter grids) and return R2 scores."""
    try:
        if param is None:
            param = {}

        report = {}

        for model_name, model in models.items():
            # If no hyperparameters provided for this model, fit directly
            if model_name not in param or not param[model_name]:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                report[model_name] = r2_score(y_test, preds)
                continue

            # Basic grid search (small grids) without importing sklearn.model_selection
            grids = param[model_name]
            keys = list(grids.keys())
            values = [grids[k] for k in keys]

            best_score = -np.inf

            def _iter(i, current):
                nonlocal best_score
                if i == len(keys):
                    # IMPORTANT: create a fresh model instance each time.
                    # Some models (e.g., CatBoost) cannot change params after fitting.
                    m = model.__class__(**model.get_params())
                    if current:
                        m.set_params(**current)
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    score = r2_score(y_test, preds)
                    if score > best_score:
                        best_score = score
                    return

                for v in values[i]:
                    current[keys[i]] = v
                    _iter(i + 1, current)

            _iter(0, {})
            report[model_name] = float(best_score)


        return report

    except Exception as e:
        raise CustomException(e, sys)

