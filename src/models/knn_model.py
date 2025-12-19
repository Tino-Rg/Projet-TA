from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors.
    Distance-based classifier. Heavily relies on Feature Scaling.
    """

    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        print(f"Training K-Nearest Neighbors (KNN)...")

        # Pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
        )

        # Hyperparameters
        param_grid = {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
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
