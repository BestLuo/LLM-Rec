# config.py
import torch

# Dataset path
DATA_PATH = 'Your Path' # Dataset folder path
TRAIN_FILE = 'TrainFiles'
TEST_FILE = 'TestFile'
COT_EMBEDDING_FILE = 'CoTFile' # CoT embedding file name

# Model hyperparameters
EMBEDDING_DIM = 64         # GNN and CoT embedding dimension
GNN_LAYERS = 3             # Number of GNN layers
TEMPERATURE_INITIAL = 0.2  # Initial temperature coefficient for contrastive loss (HGPO will adjust dynamically)
DROPOUT_RATE = 0.1         # Dropout rate for SGL graph augmentation (e.g., edge or feature dropout)

# HGPO related hyperparameters
POLICY_LEARNING_RATE = 1e-4 # Learning rate for HGPO policy network
ENTROPY_COEFF = 0.6        # Entropy regularization coefficient c1 in HGPO objective
HARMONIZATION_COEFF = 0.5   # Harmonization loss coefficient lambda_harm in HGPO objective
TEMPERATURE_REWARD_COEFF = 1.2 # HGPO temperature reward coefficient w5
PPO_CLIP_EPSILON = 0.2      # HGPO clipping parameter epsilon
NUM_NEGATIVE_SAMPLES_HGPO = 64 # Number of negative samples selected by HGPO M
NUM_GROUPS_HGPO = 5         # Number of node groups K in HGPO
GAMMA_RL = 0.99             # Discount factor in RL

# Reward function thresholds
THETA_FN = 0.8
THETA_EASY = 0.5
THETA_FP = 0.8
THETA_EASY_LOW = 0.2
REWARD_W1 = 1.0 
REWARD_W2 = 1.0 
REWARD_W3 = 1.0 
REWARD_W4 = 0.5 

# Training parameters
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3       # Learning rate for main GNN model
EPOCHS = 1000
WEIGHT_DECAY = 1e-5        # Weight decay for optimizer
K_EVAL = 20                # K value for Top-K evaluation

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SGL related (for graph augmentation)
# SGL_AUG_TYPE_1 = 'edge_drop' # First augmentation type
# SGL_AUG_TYPE_2 = 'node_drop' # Second augmentation type (or 'feature_mask', 'rw')
# SGL_AUG_RATIO_1 = DROPOUT_RATE
# SGL_AUG_RATIO_2 = DROPOUT_RATE

print(f"Device configuration: {DEVICE}")
print(f"Default data path (DATA_PATH): {DATA_PATH}")
print(f"Default CoT embedding file name (COT_EMBEDDING_FILE): {COT_EMBEDDING_FILE}")
