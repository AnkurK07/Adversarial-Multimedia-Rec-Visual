# Vission transformer for feature extraction
import requests
import numpy as np
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm  

from src.logger import logging
from src.entity.config_entity import FeatureExtractionConfig


class ImageFeatureExtractor:
    """
    A class to extract image features using a pre-trained Vision Transformer model.
    """
    def __init__(self):
        # Load the pre-trained model and feature extractor
        logging.info("Initializing the Vision Transformer model and feature extractor.")
        self.processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def extract_features(self, image_urls):
        """
        Extracts image features from a list of image URLs.

        Args:
            image_urls: A list of image URLs.

        Returns:
            A NumPy array of shape (num_images, feature_dim) containing the extracted features.
        """
        logging.info(f"Starting feature extraction for {len(image_urls)} images.")
        image_features = []
        success_count = 0  # Counter for successfully extracted features
        total_images = len(image_urls)

        # Initialize the progress bar
        with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
            for url in image_urls:
                try:
                    logging.info(f"Processing image: {url}")

                    # Load the image from URL
                    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

                    # Preprocess the image
                    inputs = self.processor(images=image, return_tensors="pt")

                    # Move tensors and model to GPU if available
                    if torch.cuda.is_available():
                        inputs = inputs.to('cuda')
                        self.model.to('cuda')

                    # Get model outputs
                    with torch.no_grad():  # Disable gradient calculation for inference
                        outputs = self.model(**inputs)

                    # Extract features (mean of the last hidden state)
                    last_hidden_states = outputs.last_hidden_state
                    feature = last_hidden_states.mean(dim=1).cpu().numpy()  # Move back to CPU for NumPy
                    image_features.append(feature)
                    success_count += 1  # Increment success count

                    logging.info(f"Successfully extracted features for image: {url}")

                except Exception as e:
                    logging.error(f"Error processing image {url}: {e}")
                    # Append a zero vector in case of error
                    image_features.append(np.zeros((1, 768)))  # Assuming 768 is the feature dimension

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(Success=success_count, Errors=pbar.n - success_count)

        # Print final summary
        logging.info(f"Total images processed: {total_images}. Features successfully extracted: {success_count}.")

        # Concatenate features and save to file
        features_array = np.concatenate(image_features, axis=0)
        logging.info(f"Saving extracted features to {FeatureExtractionConfig.feature_store_file_path}.")
        np.save(FeatureExtractionConfig.feature_store_file_path, features_array)

        return features_array

