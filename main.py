import os
import time
import warnings

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
# from src.models.knn_model import KNNModel
# from src.models.logistic_regression_model import LogisticRegressionModel
# from src.models.softmax_model import SoftmaxModel
# from src.models.svm_model import SVMModel
# from src.models.decision_tree_model import DecisionTreeModel
from src.models.mlp_model import MLPModel

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    # Load Data
    # Check if train.csv is in the data/raw/ folder
    loader = DataLoader('data/raw/train.csv')
    X, y, classes = loader.load_data()

    # Split Data (Train / Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define the list of models to test
    models_to_test = [
        # ("K-Nearest Neighbors", KNNModel()),
        # ("Logistic Regression", LogisticRegressionModel()),
        # ("Softmax", SoftmaxModel()),
        # ("Support Vector Machine", SVMModel()),
        # ("Decision Tree", DecisionTreeModel()),
        ("MLP (Neural Network)", MLPModel())
    ]

    # dataframe to store final results for comparison
    results_data = []

    # Main Loop
    for name, model in models_to_test:
        print(f"\n--- Processing: {name} ---")

        # Mesure du temps d'entraînement
        start_time = time.time()
        model.train(X_train, y_train)
        end_time = time.time()
        duration = end_time - start_time

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        prec = precision_score(
            y_test,
            predictions,
            average='macro',
            zero_division=0)
        rec = recall_score(
            y_test,
            predictions,
            average='macro',
            zero_division=0)

        print(f"-> Accuracy: {acc:.4f}")
        print(f"-> Precision: {prec:.4f}")
        print(f"-> Recall: {rec:.4f}")
        print(f"-> F1-Score : {f1:.4f}")
        print(f"-> Time     : {duration:.2f} sec")
        # Optional: print(classification_report(y_test, predictions))

        results_data.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Time (s)': duration,
            'Best Param': str(model.get_best_params())
        })

    # Final Comparison
    print("\n========================================")
    print("          FINAL RESULTS SUMMARY         ")
    print("========================================")
    df_results = pd.DataFrame(results_data)

    df_results = df_results.sort_values(by='Accuracy', ascending=False)

    print(df_results[['Model', 'Accuracy', 'Precision',
          'Recall', 'F1-Score', 'Time (s)']].to_string(index=False))

    # Save in CSV
    df_results.to_csv('resultats_finaux.csv', index=False)
    print("========================================")


if __name__ == "__main__":
    main()
