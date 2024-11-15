from models import GradientBoostedDecisionTrees
from zenml import step
from zenml.integrations import pycaret
# Orchestrate model training


class ModelTrainingStep:
    def __init__(self, train_data, model, params=None):
        self.train_data = train_data
        self.params = params
        self.model = model

    @step
    def execute(self):
        print("Hemos entrau menor")
        X_train, y_train = self.train_data
        self.model = self.model.grid_search(X_train, y_train)
        return self.model
