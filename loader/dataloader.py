import os
import time
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_and_embed_texts(folders, chunk_size=250, chunk_overlap=50, print_info=False):
    """
    Loads .txt files from a folder, splits them into chunks, and embeds them with SentenceTransformer.
    Returns a structured dataset containing embeddings and metadata.

    Args:
        folders (list of str): Path to the folder containing .txt files.
        chunk_size (int): Maximum size of each chunk (default is 500).
        chunk_overlap (int): Overlap size between chunks (default is 50).

    Returns:
        pd.DataFrame: A DataFrame containing embeddings and metadata for all chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    data = []
    
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                chunks = splitter.split_text(text)
                embeddings = embedder.encode(chunks)

                for chunk, embedding in zip(chunks, embeddings):
                    data.append({
                        "embedding": torch.tensor(embedding),
                        "chunk_text": chunk,
                        "source_file": filename
                    })
                if print_info:
                    print(f"Embedded {filename} with {len(chunks)} chunks.")
    
    return pd.DataFrame(data)


class EmbeddingDataset(Dataset):
    def __init__(self, dataframe, file_to_label, num_labels):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing embeddings and metadata.
            file_to_label (dict): Mapping from file names to numerical labels.
        """
        self.dataframe = dataframe
        self.file_to_label = file_to_label
        self.num_labels = num_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        embedding = row["embedding"]
        source_file = row["source_file"]
        label = self.file_to_label[source_file]
        return {
            "embedding": embedding, 
            "label": label
        }
    

def balance_dataset(dataframe, file_to_label, max_multiplier=1.5):
    """
    Balances the dataset by ensuring each label has between the minimum size and max_multiplier times the minimum size.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing embeddings and metadata.
        file_to_label (dict): Mapping from file names to numerical labels.
        max_multiplier (float): The maximum size multiplier for each label (default is 1.5).

    Returns:
        pd.DataFrame: Balanced dataset with controlled sampling for each label.
    """
    dataframe['label'] = dataframe['source_file'].map(file_to_label)
    min_size = dataframe.groupby('label').size().min()
    max_size = int(min_size * max_multiplier)
    
    balanced_groups = dataframe.groupby('label').apply(
        lambda group: group.sample(n=min(len(group), max_size), random_state=42),
        include_groups=False
    )
    return balanced_groups.reset_index(drop=True)


def custom_collate_fn(batch):
    """ Collate function for batching embeddings and labels. """
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return embeddings, labels


def create_custom_dataloader(dataframe, file_to_label, num_labels, batch_size=32, shuffle=True):
    """ Creates a DataLoader for the custom dataset. """
    dataset = EmbeddingDataset(dataframe, file_to_label, num_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )
    return dataloader


def get_dataloader(data_folders, hyperparams, label_mapping, balance_data=True, print_info=True):
    start = time.time()
    dataframe = load_and_embed_texts(data_folders, chunk_size=hyperparams['chunk_size'], chunk_overlap=hyperparams['chunk_overlap'], print_info=print_info)
    if balance_data:
        dataframe = balance_dataset(dataframe, label_mapping, max_multiplier=hyperparams['balance_multiplier'])
    
    dataloader = create_custom_dataloader(dataframe, label_mapping, num_labels=hyperparams['num_labels'], batch_size=hyperparams['batch_size'], shuffle=True)
    duration = time.time() - start
    
    print(f"Dataloader from ('{data_folders}') created with {len(dataloader)} embeddings in {duration:.1f} seconds.")
    return dataloader