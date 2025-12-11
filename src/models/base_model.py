from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class defining the standard interface for all
    classification models in this project.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the model using Cross-Validation and GridSearch for hyperparameter tuning.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts classes for the test set.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted labels.
        """
        pass

    @abstractmethod
    def get_best_params(self) -> dict:
        """
        Returns the optimal hyperparameters found during training.
        """
        pass
