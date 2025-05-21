# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

from lghrec_project.config import DATA_PATH, TRAIN_FILE, TEST_FILE, COT_EMBEDDING_FILE, EMBEDDING_DIM

class RecommendationDataset(Dataset):
    def __init__(self, file_path, user_map, item_map, cot_embeddings_map=None, item_id_key='asin'):
        """
        Initialize recommendation dataset.
        Args:
            file_path (str): Data file path (e.g. train.csv, test.csv).
            user_map (dict): Mapping from original user IDs to consecutive integer IDs.
            item_map (dict): Mapping from original item IDs to consecutive integer IDs.
            cot_embeddings_map (dict, optional): Mapping from original item IDs to CoT embeddings.
            item_id_key (str): Key name for item ID in CoT embedding JSON file.
        """
        self.df = pd.read_csv(file_path, header=None, names=['user_orig_id', 'item_orig_id'])
        
        # Map original IDs to consecutive integer IDs
        self.df['user_id'] = self.df['user_orig_id'].map(user_map)
        self.df['item_id'] = self.df['item_orig_id'].map(item_map)

        # Filter out unmapped IDs
        self.df.dropna(subset=['user_id', 'item_id'], inplace=True)
        self.df['user_id'] = self.df['user_id'].astype(int)
        self.df['item_id'] = self.df['item_id'].astype(int)

        self.users = torch.tensor(self.df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(self.df['item_id'].values, dtype=torch.long)
        
        self.cot_embeddings_map = cot_embeddings_map
        self.item_id_key = item_id_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        return user, item

def load_cot_embeddings(json_file_path, item_map, item_id_key='asin', embedding_dim=EMBEDDING_DIM):
    """
    Load CoT embeddings and map to consecutive item IDs.
    Args:
        json_file_path (str): CoT embedding JSON file path.
        item_map (dict): Mapping from original item IDs to consecutive integer IDs.
        item_id_key (str): Key name for item ID in JSON.
        embedding_dim (int): Embedding dimension.
    Returns:
        torch.Tensor: CoT embedding matrix, shape (num_mapped_items, embedding_dim).
                      Index corresponds to consecutive ID in item_map.
                      If an ID has no CoT embedding, use zero vector.
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: CoT embedding file {json_file_path} not found. Returning zero embeddings.")
        max_item_id_val = 0
        if item_map:
            max_item_id_val = max(item_map.values()) if item_map else -1

        return torch.zeros((max_item_id_val + 1, embedding_dim), dtype=torch.float)

    with open(json_file_path, 'r') as f:
        raw_cot_data = json.load(f)

    if not item_map:
        max_raw_id = 0
        for entry in raw_cot_data:
            if isinstance(entry.get(item_id_key), int) and entry.get(item_id_key) > max_raw_id:
                 max_raw_id = entry.get(item_id_key)
        cot_embeddings_tensor = torch.zeros((max_raw_id + 1, embedding_dim), dtype=torch.float)
        
        for entry in raw_cot_data:
            raw_item_id = entry.get(item_id_key)
            embedding = entry.get('embedding')
            if isinstance(raw_item_id, int) and embedding:
                if raw_item_id < len(cot_embeddings_tensor):
                     cot_embeddings_tensor[raw_item_id] = torch.tensor(embedding, dtype=torch.float)
                else:
                    print(f"Warning: Original item ID {raw_item_id} in CoT file exceeds inferred range.")
        return cot_embeddings_tensor

    # Build based on item_map
    max_mapped_item_id = 0
    if item_map:
        max_mapped_item_id = max(item_map.values()) if item_map else -1

    cot_embeddings_tensor = torch.zeros((max_mapped_item_id + 1, embedding_dim), dtype=torch.float)
    
    found_embeddings = 0
    for entry in raw_cot_data:
        raw_item_id = entry.get(item_id_key) # This is the original ID
        embedding = entry.get('embedding')

        if raw_item_id is not None and embedding:
            # Check if original ID is in item_map
            if raw_item_id in item_map:
                mapped_item_id = item_map[raw_item_id] # Get mapped consecutive ID
                cot_embeddings_tensor[mapped_item_id] = torch.tensor(embedding, dtype=torch.float)
                found_embeddings += 1

    print(f"Loaded embeddings for {found_embeddings} items from CoT file.")
    return cot_embeddings_tensor


def create_mappings_and_load_data(data_path, train_file, test_file, cot_embedding_file, item_id_key_in_cot='asin'):
    """
    Create user and item ID mappings and load train/test datasets.
    """
    train_df_path = os.path.join(data_path, train_file)
    test_df_path = os.path.join(data_path, test_file)
    cot_json_path = os.path.join(data_path, cot_embedding_file)
    train_df = pd.read_csv(train_df_path, header=None, names=['user_orig_id', 'item_orig_id'])
    test_df = pd.read_csv(test_df_path, header=None, names=['user_orig_id', 'item_orig_id'])
    
    # Merge all users and items from train and test sets to create full mapping
    all_users_orig = pd.concat([train_df['user_orig_id'], test_df['user_orig_id']]).unique()
    all_items_orig = pd.concat([train_df['item_orig_id'], test_df['item_orig_id']]).unique()

    user_map = {orig_id: i for i, orig_id in enumerate(all_users_orig)}
    item_map = {orig_id: i for i, orig_id in enumerate(all_items_orig)} # item_map key is original ID

    num_users = len(user_map)
    num_items = len(item_map)
    
    print(f"Total unique users: {num_users}")
    print(f"Total unique items: {num_items}")

    # Load CoT embeddings
    cot_embeddings_tensor = load_cot_embeddings(cot_json_path, item_map, item_id_key=item_id_key_in_cot)
    
    train_dataset = RecommendationDataset(train_df_path, user_map, item_map)
    test_dataset = RecommendationDataset(test_df_path, user_map, item_map)

    # Get all interaction history for evaluation exclusion
    all_interactions_df = pd.concat([train_df, test_df])
    user_history = defaultdict(set)
    for _, row in all_interactions_df.iterrows():
        u_orig, i_orig = row['user_orig_id'], row['item_orig_id']
        if u_orig in user_map and i_orig in item_map:
             user_history[user_map[u_orig]].add(item_map[i_orig])

    return train_dataset, test_dataset, num_users, num_items, user_map, item_map, cot_embeddings_tensor, user_history

