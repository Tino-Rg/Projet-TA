from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression implementation with hyperparameter tuning
    and automatic feature scaling.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        print(f"Training Logistic Regression...")

        # Create Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                max_iter=5000,    # Increased to prevent convergence warnings
                tol=1e-3,
                solver='liblinear',
                random_state=self.random_state
            ))
        ])

        # Define Hyperparameters for GridSearch
        param_grid = {
            'logreg__C': [0.01, 0.1, 1, 10, 100],
            'logreg__penalty': ['l1', 'l2']
        }

        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,               # 5-Fold Cross-Validation
            scoring='accuracy',
            n_jobs=1,
            verbose=1
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
