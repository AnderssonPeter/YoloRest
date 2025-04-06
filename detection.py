from pydantic import BaseModel
from label import Label

# Array with [class_id, score, box[0], box[1], box[2], box[3]]
class Detection(BaseModel):
    label: Label
    score: float
    top: float
    left: float
    bottom: float
    right: float