# Adversial Multimedia Recommendation configuration

from src.logger import logging
import cornac
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit
from src.Adversial_Multimedia_Recommendation.AMRParameters import AMRParameters

class AMR:
    def __init__(self, image_features, user_item, item_ids,  params: AMRParameters = None):
        """
        Initialize the ARM class for adversarial multimedia recommendation.

        Parameters:
        - image_features: numpy array of extracted image features
        - user_item: user-item interaction data (e.g., list of tuples (user, item, rating))
        - item_ids: list of item IDs corresponding to the image features
        - params: ARMParameters dataclass instance containing model configurations
        """
        self.image_features = image_features
        self.user_item = user_item
        self.item_ids = item_ids
        
         # Use default parameters if none are provided
        self.params = params if params is not None else AMRParameters()

        # Setup logger
        self.logger = logging.getLogger("ARM")
        self.logger.info("ARM class initialized.")

        # Create the ImageModality instance
        self.logger.info("Setting up ImageModality.")
        self.item_image_modality = ImageModality(
            features=self.image_features,
            ids=self.item_ids,
            normalized=True
        )

        # Create the RatioSplit instance
        self.logger.info("Setting up RatioSplit.")
        self.ratio_split = RatioSplit(
            data=self.user_item,
            test_size=self.params.test_size,
            rating_threshold=self.params.rating_threshold,
            exclude_unknowns=True,
            verbose=self.params.split_verbose,
            item_image=self.item_image_modality,
        )

        # Instantiate the AMR model
        self.logger.info("Instantiating AMR model.")
        self.amr = cornac.models.AMR(
            k=self.params.k,
            k2=self.params.k2,
            n_epochs=self.params.n_epochs,
            batch_size=self.params.batch_size,
            learning_rate=self.params.learning_rate,
            lambda_w=self.params.lambda_w,
            lambda_b=self.params.lambda_b,
            lambda_e=self.params.lambda_e,
            use_gpu=self.params.use_gpu,
            trainable=self.params.trainable,
            verbose=self.params.verbose,
            init_params=self.params.init_params,
            seed=self.params.seed
        )
        self.logger.info("AMR model instantiated.")

    def get_model(self):
        """
        Returns the instantiated AMR model.
        """
        return self.amr

    def get_ratio_split(self):
        """
        Returns the RatioSplit instance used for evaluation.
        """
        return self.ratio_split
