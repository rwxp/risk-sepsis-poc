# -*- coding: utf-8 -*-
"""Nuevos modelos con GridSearch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XINXxF73AW6QVvZr6Wbgi8R5dl-KMQar

# Librerias
"""

# from sklearn.metrics import accuracy_score, roc_auc_score
# from xgboost.sklearn import XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.preprocessing import label_binarize
# import lightgbm as lgb
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import seaborn as sn
# import pandas as pd
# from math import e
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn import linear_model
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import preprocessing
# from sklearn.model_selection import StratifiedShuffleSplit
# from mlxtend.plotting import plot_confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import RepeatedStratifiedKFold
# import joblib


import joblib
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import label_binarize


class GradientBoostedDecisionTrees:
    def __init__(self, cross_validation, n_estimators=19, random_state=2016, min_samples_leaf=8):
        # Inicializa el modelo con los hiperparámetros por defecto
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state, min_samples_leaf=min_samples_leaf
        )
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.best_model = None
        self.best_params = None
        self.cross_validation = cross_validation

    def grid_search(self, X_train, y_train, n_splits=5, n_repeats=3, random_state=1):
        """
        Realiza la búsqueda de hiperparámetros mediante GridSearchCV.
        """

        # Define el espacio de búsqueda
        space = dict()
        space['max_features'] = ['sqrt', 'log2']
        space['max_depth'] = [None, 1, 3, 5, 10, 20]
        space['subsample'] = [0.5, 1]
        space['learning_rate'] = [0.001, 0.01, 0.1]

        # Realiza la búsqueda de hiperparámetros
        search = GridSearchCV(
            self.model, space, scoring='accuracy', n_jobs=-1, cv=self.cross_validation)
        result = search.fit(X_train, y_train)
        # Almacena el mejor modelo y parámetros
        self.best_model = result
        self.best_params = result.best_params_
        return self.best_model, self.best_params

    def predict(self, X_test):
        """
        Realiza la predicción usando el mejor modelo encontrado.
        """
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        y_pred = self.best_model.predict(X_test)
        print("setiembre", y_pred)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo utilizando la curva ROC.
        """
        y_test_hot = label_binarize(y_test, classes=(0, 1))
        y_score = self.best_model.decision_function(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_hot.ravel(), y_score.ravel())

        print("FPR: ", fpr)
        print("TPR: ", tpr)

        return fpr, tpr, thresholds

    def save_model(self, filename='ModeloG1_GBDT.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar.")

        joblib.dump(self.best_model, filename)
        print(f'Modelo guardado en: {filename}')
