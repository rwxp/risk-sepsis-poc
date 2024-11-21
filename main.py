from pipelines import DataPreprocessingStep
from pipelines import ModelTrainingStep
from pipelines import Metrics
from models import GradientBoostedDecisionTrees
from zenml import pipeline
import os
import joblib


@pipeline
def main():
    # Process and divide initial dataset
    data_processor = DataPreprocessingStep(
        input_path='data/raw/SepsisTraining.DataPacientes6-8.csv')
    X_train, X_test, y_train, y_test = data_processor.preprocess_data()
    print("Hemos entrau")
    cross_validation = data_processor.cross_validation()
    # Train specific model
    algorithm = GradientBoostedDecisionTrees(cross_validation=cross_validation)
    model_trainer = ModelTrainingStep(
        train_data=(X_train, y_train), algorithm=algorithm)

    model_trainer.train()

    # y_pred = model.predict(X_test)
    # model_validator = Metrics(
    #     y_pred,  X_train, y_train, X_test, y_test)

    # model_validator.plot_f1_score()
    # model_validator.plot_confusion_matrix()
