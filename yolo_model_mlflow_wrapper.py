import base64
import io
from typing import Any, Dict

import cv2
import mlflow.pyfunc
import numpy as np
from PIL import Image
from ultralytics import YOLO


def base64_to_ndarray(base64_str):
    img_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_data))
    return np.array(pil_image.convert("RGB"), dtype=np.uint8)


class MLFlowYOLO(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def load_context(self, context):
        print("load_context")
        self.my_model = YOLO(context.artifacts["model"])
        print("load_context")

    def predict(self, context, model_input, params):
        print("predict")
        if isinstance(model_input, str):
            img = base64_to_ndarray(model_input)
        elif isinstance(model_input, np.ndarray):
            print(model_input.dtype)
            if model_input.dtype == "str":
                img = base64_to_ndarray(model_input)
            else:
                img = np.array(model_input, dtype=np.uint8)
        else:
            raise ValueError("Input must be image path or numpy array")

        pred = self.my_model.predict(img)[0]
        predictions = []

        for cord, c, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
            predictions.append(
                {
                    "bbox": [int(x) for x in cord],
                    "confidence": float(conf),
                    "class": int(c),
                    "class_name": pred.names[int(c)],
                }
            )

        return predictions
