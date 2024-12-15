import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Load the trained model from the BentoML model store
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Create a BentoML service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    # Use the model to predict the class for the input data
    result = iris_clf_runner.predict.run(input_series)
    return result
