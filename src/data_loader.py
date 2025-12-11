import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """
    Responsible for loading data and preparing it for training.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.le = LabelEncoder()

    def load_data(self):
        """
        Loads the CSV, separates features and target, and encodes the target.

        Returns:
            X (DataFrame): Features (margin, shape, texture)
            y (Series): Encoded Target (species)
            classes (array): Original class names (for reference)
        """
        df = pd.read_csv(self.file_path)

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # Separate Features (X) and Target (y)
        X = df.drop(columns=['species'])
        y_raw = df['species']

        # Encode target labels
        y = self.le.fit_transform(y_raw)

        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
        return X, y, self.le.classes_
