from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """
    Decision Tree (CART Algorithm).
    Non-parametric supervised learning method.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        print(f"Training Decision Tree (CART)...")

        # Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('dt', DecisionTreeClassifier(random_state=self.random_state))
        ])

        # Hyperparamètres
        param_grid = {
            'dt__criterion': ['gini', 'entropy'],
            'dt__max_depth': [None, 10, 20, 30, 50],
            'dt__min_samples_split': [2, 5, 10],
            'dt__min_samples_leaf': [1, 2, 4]
        }

        # Grid Search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
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
