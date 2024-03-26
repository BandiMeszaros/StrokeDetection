from pydantic import BaseModel
class PredictSchema(BaseModel):
    prediction: dict
    target_column: int

class TrainSchema(BaseModel):
    GradientBoosting: list
    KNN: list
    RandomForest: list
    SVM: list
