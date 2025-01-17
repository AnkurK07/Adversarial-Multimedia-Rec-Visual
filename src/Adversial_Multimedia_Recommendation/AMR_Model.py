"""Adversial Multimedia Recommendation Model For Uses"""
import pickle
import mlflow
import mlflow.pyfunc
import cornac




class AMRModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None, model_path=None):
        if model is not None:
            # If model is provided directly, use it
            self.model = model
        elif model_path is not None:
            # If model path is provided, load the model
            self.model = cornac.models.AMR.load(model_path)
        else:
            raise ValueError("Either 'model' or 'model_path' must be provided.")

    def predict(self, context, model_input):
        # model_input should be a tuple (user_id, k)
        user_id, k = model_input
        # Get the recommendations for the given user_id and k
        recommendations = self.model.recommend(user_id, k=k)
        return recommendations

    def save_model(self, filepath):
        """Save the entire AMRWrapper (including the model) to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        """Load the AMRWrapper from a pickle file."""
        with open(filepath, "rb") as f:
            wrapper = pickle.load(f)
        return wrapper
