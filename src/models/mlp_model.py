from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron (MLP).
    Artificial Neural Network.
    Requires strictly scaled data (StandardScaler).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        print(f"Training Multi-Layer Perceptron (Neural Network)...")

        # Pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=1000,
                        early_stopping=True,  # Stop if it doesn't improve
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # Hyperparameters
        # hidden_layer_sizes:
        #   (100,) = 1 hidden layer with 100 neurons.
        #   (100, 50) = 2 hidden layers (Deep Learning).
        # solver:
        #   'adam' is standard for big data.
        #   'lbfgs' is often BETTER and FASTER for small datasets (<2000 samples).
        param_grid = {
            "mlp__hidden_layer_sizes": [(100,), (100, 50)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__solver": ["adam", "lbfgs"],
            "mlp__alpha": [0.0001, 0.01],  # L2 Regularization to prevent overfitting
        }

        # Grid Search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Best Params: {grid_search.best_params_}")

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        return self.model.predict(X_test)

    def get_best_params(self):
        return self.best_params
