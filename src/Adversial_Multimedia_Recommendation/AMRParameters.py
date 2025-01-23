# Parameters for AMR Model
from dataclasses import dataclass

@dataclass
class AMRParameters:
    k: int = 50                    # The dimension of the gamma latent factors.
    k2: int = 50                    # The dimension of the theta latent factors.
    n_epochs: int = 5            # Maximum number of epochs for SGD.
    batch_size: int = 1024          # The batch size for SGD.
    learning_rate: float = 0.001    # The learning rate for SGD.
    lambda_w: float = 1             # Regularization for latent factor weights.
    lambda_b: float = 0.01          # Regularization for biases.
    lambda_e: float = 0.01          # Regularization for embedding matrix E and beta prime vector.
    use_gpu: bool = True            # Whether or not to use GPU.
    trainable: bool = True          # When False, assumes the model is pre-trained.
    verbose: bool = False           # When True, running logs are displayed.
    init_params: dict = None        # Initial parameters (optional).
# Ratio Split Parameters    
    seed: int = None                # Random seed for weight initialization.
    test_size: float = 0.1          # Test set ratio for splitting.
    rating_threshold: float = 0.5   # Rating threshold for interactions.
    split_verbose: bool = False     # Verbosity for the RatioSplit.
    K = 50                          # The number of items in the top@k list
