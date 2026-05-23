import os
import sys
from dataclasses import dataclass

import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """Holds configuration for model training."""

    # Keep artifact naming consistent
    model_obj_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        """Train a regression model and print accuracy-like metric.

        This project uses a simple regression setup so the script prints a
        score to the console.
        """

        try:
            # train_arr/test_arr are expected to be:
            # [preprocessed features..., target]
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Use a lightweight sklearn model
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score

            regressor = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )

            regressor.fit(X_train, y_train)
            preds = regressor.predict(X_test)

            r2 = float(r2_score(y_test, preds))

            save_object(
                file_path=self.model_trainer_config.model_obj_file_path,
                obj=regressor,
            )

            logging.info("Model training completed. R2 score: %s", r2)

            # IMPORTANT: print something to CMD (user expects it)
            return {"r2_score": r2, "model_path": self.model_trainer_config.model_obj_file_path}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # For manual testing
    pass

