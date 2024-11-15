import pandas as pd
import numpy as np
from typing_extensions import Annotated
from typing_extensions import Tuple

from utils import shuffle_split, repeated_stratified_k_fold
from zenml import step

# Se debe cambiar nombre  a PreProcessing


class DataPreprocessingStep:
    def __init__(self, input_path, n_splits=5, n_repeats=3, random_state=1):
        self.input_path = input_path
        self.df = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        # half_length = len(self.df) // 16
        # self.df = self.df.iloc[:half_length]

    # Organize the dataframe and split the dataset into training and testing sets.

    @step
    def preprocess_data(self) -> Tuple[
        Annotated[np.ndarray, "X_train"],
        Annotated[np.ndarray, "X_test"],
        Annotated[np.ndarray, "y_train"],
        Annotated[np.ndarray, "y_test"]
    ]:
        self.load_data()
        X_train, X_test, y_train, y_test = shuffle_split(self.df)

        return X_train, X_test, y_train, y_test

    def cross_validation(self):
        cross_validation = repeated_stratified_k_fold(
            self.n_splits, self.n_repeats, self.random_state)
