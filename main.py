# main.py
import torch
from torch.utils.data import DataLoader
import os
import argparse

from lghrec_project.config import (DEVICE, DATA_PATH as DEFAULT_DATA_PATH, 
                    TRAIN_FILE as DEFAULT_TRAIN_FILE,
                    TEST_FILE as DEFAULT_TEST_FILE,
                    COT_EMBEDDING_FILE as DEFAULT_COT_FILE,
                    EMBEDDING_DIM, GNN_LAYERS, TEMPERATURE_INITIAL, DROPOUT_RATE,
                    POLICY_LEARNING_RATE, ENTROPY_COEFF, HARMONIZATION_COEFF,
                    PPO_CLIP_EPSILON, NUM_NEGATIVE_SAMPLES_HGPO, NUM_GROUPS_HGPO,
                    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, K_EVAL)

from data.dataset import create_mappings_and_load_data
from data.graph_utils import build_dgl_graph, get_graph_node_degrees
from models.lghrec_model import LGHRec
from rl.hgpo_agent import HGPOAgent
from lghrec_project.trainer import Trainer
from utils.logger import setup_logger
from datetime import datetime
import numpy as np 

def main(args):
    # --- 1. Set up logging ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(log_dir, f"lghrec_run_{current_time}_{args.dataset_name}.log")
    logger = setup_logger(log_file=log_file)
    logger.info(f"Experiment start time: {current_time}")
    logger.info(f"Device used: {DEVICE}")
    logger.info(f"Parameter config: {args}")
    logger.info(f"Training file used: {os.path.join(args.data_path, args.train_file)}")
    logger.info(f"Test file used: {os.path.join(args.data_path, args.test_file)}")
    logger.info(f"CoT embedding file used: {os.path.join(args.data_path, args.cot_file)}")

    # --- 2. Load and preprocess data ---
    logger.info("Start loading and preprocessing data...")
    
    item_id_key_for_cot = 'asin' # Key name for item ID in CoT embedding file

    train_dataset, test_dataset, num_users, num_items, \
    user_map, item_map, cot_embeddings_tensor, user_history = \
        create_mappings_and_load_data(
            data_path=args.data_path,       
            train_file=args.train_file,     
            test_file=args.test_file,       
            cot_embedding_file=args.cot_file, 
            item_id_key_in_cot=item_id_key_for_cot
        )
    logger.info(f"Data loaded: num_users={num_users}, num_items={num_items}")
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"CoT embedding tensor shape: {cot_embeddings_tensor.shape}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 4, shuffle=False, num_workers=args.num_workers)

    # --- 3. Build DGL graph and get node degrees ---
    logger.info("Start building DGL graph...")
    dgl_graph = build_dgl_graph(num_users, num_items, train_dataset)
    user_degrees, item_degrees = get_graph_node_degrees(dgl_graph, num_users, num_items)
    logger.info("DGL graph built, node degrees calculated.")
    logger.info(f"Average user degree: {user_degrees.float().mean().item():.2f}, Average item degree: {item_degrees.float().mean().item():.2f}")

    # --- 4. Initialize model ---
    logger.info("Start initializing LGHRec model and HGPO Agent...")
    lghrec_model = LGHRec(
        num_users=num_users,
        num_items=num_items,
        gnn_embedding_dim=args.embed_dim,
        cot_embedding_dim=cot_embeddings_tensor.shape[1],
        dgl_graph=dgl_graph,
        initial_cot_embeddings_tensor=cot_embeddings_tensor,
        sgl_temp=args.sgl_temp,
        sgl_dropout=args.sgl_dropout,
        num_gnn_layers=args.gnn_layers,
        device=DEVICE
    ).to(DEVICE)

    rl_state_dim = args.embed_dim * 3 + 1
    
    hgpo_agent = HGPOAgent(
        state_dim=rl_state_dim,
        num_candidate_neg_samples=args.rl_candidate_pool_size,
        num_groups=args.rl_num_groups,
        device=DEVICE
    )
    logger.info("Model and Agent initialized.")

    # --- 5. Initialize trainer and start training ---
    logger.info("Start initializing trainer...")
    all_item_ids_list = sorted(list(item_map.values()))

    trainer = Trainer(
        lghrec_model=lghrec_model,
        hgpo_agent=hgpo_agent,
        train_loader=train_loader,
        test_loader=test_loader,
        num_users=num_users,
        num_items=num_items,
        all_item_ids_for_eval=all_item_ids_list,
        user_degrees_tensor=user_degrees,
        item_degrees_tensor=item_degrees,
        user_history_for_eval=user_history,
        logger=logger,
        device=DEVICE
    )
    logger.info("Trainer initialized. Start training...")
    
    trainer.run()

    logger.info("Experiment finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LGHRec: LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization")
    
    # Data and path parameters
    # Use values from config.py as argparse defaults
    parser.add_argument('--dataset_name', type=str, default='dataset_name', help='Dataset name')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, 
                        help=f'Base path for dataset files (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--train_file', type=str, default=DEFAULT_TRAIN_FILE, 
                        help=f'Training data file name (under data_path) (default: {DEFAULT_TRAIN_FILE})')
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST_FILE, 
                        help=f'Test data file name (under data_path) (default: {DEFAULT_TEST_FILE})')
    parser.add_argument('--cot_file', type=str, default=DEFAULT_COT_FILE, 
                        help=f'CoT embedding file name (under data_path) (default: {DEFAULT_COT_FILE})')
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=EMBEDDING_DIM, help='GNN embedding dimension')
    parser.add_argument('--gnn_layers', type=int, default=GNN_LAYERS, help='Number of GNN layers')
    parser.add_argument('--sgl_temp', type=float, default=TEMPERATURE_INITIAL, help='Initial temperature for SGL contrastive loss')
    parser.add_argument('--sgl_dropout', type=float, default=DROPOUT_RATE, help='Dropout rate for SGL graph augmentation')

    # HGPO Agent parameters
    parser.add_argument('--rl_candidate_pool_size', type=int, default=100, help='Candidate pool size for negative samples in HGPO policy network')
    # NUM_NEGATIVE_SAMPLES_HGPO (actual M selected) is read from config.py, used in trainer.py
    parser.add_argument('--rl_num_groups', type=int, default=NUM_GROUPS_HGPO, help='Number of node groups K in HGPO')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Main model learning rate')
    parser.add_argument('--wd', type=float, default=WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--k_eval', type=int, default=K_EVAL, help='K value for Top-K evaluation')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
