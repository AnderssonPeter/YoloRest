from pydantic import BaseModel

# Array with [class_id, score, box[0], box[1], box[2], box[3]]
class Prediction(BaseModel):
    label: str
    confidence: float
    y_min: float
    x_min: float
    y_max: float
    x_max: float

class Predictions(BaseModel):
    predictions: list[Prediction]
    success: bool