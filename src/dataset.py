"""
Dataset classes for medical relation extraction.

This module provides dataset classes for loading and processing medical text data
for named entity recognition and relation extraction tasks.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from scipy.sparse import csr_matrix


class NERDataset(Dataset):
    """
    Dataset for Named Entity Recognition and Relation Extraction.
    
    This dataset processes medical documents with entity and relation annotations
    for training relation extraction models.
    """
    def __init__(self, docs, tokenizer, label_map, use_common_patterns=False, relation_map=None):
        """
        Initialize the dataset.
        
        Args:
            docs: List of medkit TextDocument objects
            tokenizer: Tokenizer (BERT, ClinicalBERT, etc.)
            label_map: Mapping from entity labels to IDs
            use_common_patterns: Whether to use common relation patterns
            relation_map: Mapping from relation labels to IDs
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.use_common_patterns = use_common_patterns
        self.relation_map = relation_map
        self.docs = docs  # Store the documents list
        self.data = []

        for doc in docs:
            text = doc.text
            # Extract entities with their spans, label, and ID
            entities = [
                (ent.spans[0].start, ent.spans[0].end, ent.label, ent.uid) 
                for ent in doc.anns.get_entities()
            ]
            # Extract relations with source, target, and relation type ID
            relations = [
                (rel.source_id, rel.target_id, relation_map[rel.label])  
                for rel in doc.anns.get_relations()
                if rel.label in relation_map  # Skip relations not in map
            ] 

            # Align tokens with entity labels and create relation matrices
            input_ids, attention_mask, labels, relations_matrix, ent_ids = self._align_labels(
                text, entities, relations
            )
            self.data.append((input_ids, attention_mask, labels, relations_matrix, ent_ids))

    def _align_labels(self, text, entities, relations):
        """
        Align text tokens with entity labels and create relation matrices.
        
        Args:
            text: Document text
            entities: List of (start, end, label, id) tuples
            relations: List of (source_id, target_id, relation_type) tuples
            
        Returns:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
            labels: Entity labels for each token
            relations_matrix: Relation type matrix
            ent_ids: Entity IDs for each token
        """
        # Tokenize the input text
        tokenized_inputs = self.tokenizer(text, return_offsets_mapping=True, truncation=True, padding=False)
        offsets = tokenized_inputs.pop("offset_mapping")
        word_ids = tokenized_inputs.word_ids()
        
        # Initialize labels and entity IDs
        labels = ["O"] * len(offsets)
        ent_ids = [None] * len(offsets)
        
        # Assign labels to tokens based on entity spans
        for start, end, label, ent_id in entities:
            ent_encours = None
            for idx, (offset_start, offset_end) in enumerate(offsets):
                word_idx = word_ids[idx]
                
                if word_idx is None:
                    continue  # Skip special tokens
                
                # Check if token is inside entity span
                if offset_start >= start and offset_end <= end and offset_start < offset_end + 1:
                    if offset_start == start:
                        if ent_encours is None:
                            # Beginning of entity
                            labels[idx] = f"B-{label}"
                            ent_ids[idx] = ent_id
                            ent_encours = ent_id
                        else:
                            # Continuing an entity
                            labels[idx] = f"I-{label}"
                    else:
                        # Inside an entity
                        labels[idx] = f"I-{label}"
                # Handle partial overlap
                elif start < offset_end and end > offset_start and offset_start < offset_end + 1:
                    if offset_start == start:
                        if ent_encours is None:
                            labels[idx] = f"B-{label}"
                            ent_ids[idx] = ent_id
                            ent_encours = ent_id
                        else:
                            labels[idx] = f"I-{label}"
                    else:
                        labels[idx] = f"I-{label}"
        
        # Convert string labels to IDs
        label_ids = [self.label_map.get(label, 0) for label in labels]

        # Create relation matrix
        relations = set(relations)
        relation_dict = {}
        
        for x, y, z in relations:
            if x + y not in relation_dict:
                relation_dict[x + y] = []
            relation_dict[x + y].append(z)
            
        # Check for multi-labeled relations
        for x in relation_dict.values():
            if len(x) > 1:
                print(f"Warning: Multiple relation types between same entity pair: {x}")
        
        # Create sparse relation matrix
        relation_matrix = self._create_relation_matrix(ent_ids, relations, use_common_patterns=self.use_common_patterns)

        return (
            torch.tensor(tokenized_inputs["input_ids"]),
            torch.tensor(tokenized_inputs["attention_mask"]),
            torch.tensor(label_ids),
            relation_matrix,
            ent_ids
        )

    def _create_relation_matrix(self, ent_ids, relations, use_common_patterns=False):
        """
        Create a sparse relation matrix from entity IDs and relations.
        
        Args:
            ent_ids: Entity IDs for each token
            relations: List of (source_id, target_id, relation_type) tuples
            use_common_patterns: Whether to use common relation patterns
            
        Returns:
            Relation matrix as a dense tensor
        """
        n = len(ent_ids)
        
        # Create a mapping from entity IDs to token indices
        ent_to_idx = {eid: idx for idx, eid in enumerate(ent_ids) if eid is not None}
        
        # Initialize sparse matrix components
        rows, cols, data = [], [], []
        
        # Handle multi-labeled relations (keep highest priority)
        seen = {}
        for x, y, z in relations:
            if (x, y) in seen:
                # Keep the relation with the higher ID (assuming higher ID = higher priority)
                seen[(x, y)] = (x, y, max(z, seen[(x, y)][2]))
            else:
                # Add the relation
                seen[(x, y)] = (x, y, z)

        relations = list(seen.values())
        
        # Add relations to sparse matrix
        for e1, e2, rel_class in relations:
            if e1 in ent_to_idx and e2 in ent_to_idx:
                i, j = ent_to_idx[e1], ent_to_idx[e2]
                rows.append(i)
                cols.append(j)
                data.append(rel_class)

        # Create sparse matrix
        if not rows:  # Handle case with no relations
            sparse_matrix = csr_matrix(([], ([], [])), shape=(n, n))
        else:
            sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Convert to dense tensor
        matrix = torch.sparse_csr_tensor(
            torch.tensor(sparse_matrix.indptr),
            torch.tensor(sparse_matrix.indices),
            torch.tensor(sparse_matrix.data, dtype=torch.long),
            size=(n, n)
        ).to_dense()
        
        # Set -100 for diagonal and None entities
        matrix.fill_diagonal_(-100)
        none_mask = torch.tensor([id is None for id in ent_ids])
        matrix[none_mask, :] = -100
        matrix[:, none_mask] = -100

        return matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataLoader(DataLoader):
    """
    Custom data loader for batching NER dataset instances.
    
    This data loader handles variable-length sequences and properly
    pads inputs, labels, and relation matrices.
    """
    def __init__(self, *args, max_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = lambda batch: custom_collate_fn(batch, max_length)


def custom_pad_sequence(sequences, pad_length, batch_first=True, padding_value=0):
    """
    Pad a list of variable length tensors to a fixed length.
    
    Args:
        sequences: List of tensors
        pad_length: Target length to pad to
        batch_first: Whether the output has batch size as first dimension
        padding_value: Value to use for padding
        
    Returns:
        Tensor of shape (batch_size, pad_length, ...)
    """
    # First, pad to the longest sequence in the batch
    padded = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
    
    # Get current sequence length
    current_length = padded.size(1) if batch_first else padded.size(0)
    
    # If current_length is less than pad_length, add more padding
    if current_length < pad_length:
        if batch_first:
            padding = torch.full(
                (padded.size(0), pad_length - current_length, *padded.size()[2:]), 
                padding_value, 
                dtype=padded.dtype, 
                device=padded.device
            )
            padded = torch.cat([padded, padding], dim=1)
        else:
            padding = torch.full(
                (pad_length - current_length, padded.size(1), *padded.size()[2:]), 
                padding_value, 
                dtype=padded.dtype, 
                device=padded.device
            )
            padded = torch.cat([padded, padding], dim=0)
    
    # If current_length is more than pad_length, truncate
    elif current_length > pad_length:
        if batch_first:
            padded = padded[:, :pad_length]
        else:
            padded = padded[:pad_length]
    
    return padded


def cat_matrix(all_relation_matrices, max_length):
    """
    Combine relation matrices into a single batch matrix.
    
    Args:
        all_relation_matrices: List of relation matrices
        max_length: Maximum sequence length
        
    Returns:
        Combined relation matrix, pair mask, and sentence mask
    """
    final_matrix = torch.full((max_length, max_length), -100)
    final_matrix2 = torch.full((max_length, max_length), 0)
    current_idx = 0
    
    for matrix in all_relation_matrices:
        size = matrix.size(0)
        
        if current_idx + size <= max_length:
            final_matrix[current_idx:current_idx+size, current_idx:current_idx+size] = matrix
            final_matrix2[current_idx:current_idx+size, current_idx:current_idx+size] = 1
        else:
            final_matrix[current_idx:max_length, current_idx:max_length] = matrix[:max_length-current_idx, :max_length-current_idx]
            final_matrix2[current_idx:max_length, current_idx:max_length] = 1

        current_idx += size
           
    # Create pair mask
    pair_masks_padded = create_pair_mask(final_matrix)

    return final_matrix, pair_masks_padded, final_matrix2


def create_pair_mask(entity_matrix):
    """
    Create a mask for valid entity pairs.
    
    Args:
        entity_matrix: Matrix of entity relations
        
    Returns:
        Boolean mask for valid pairs
    """
    pair_mask = (entity_matrix != -100).float()
    return pair_mask


def custom_collate_fn(batch, max_length):
    """
    Custom collate function for batching NER dataset instances.
    
    Args:
        batch: List of (input_ids, attention_mask, labels, relation_matrix, ids_labels) tuples
        max_length: Maximum sequence length
        
    Returns:
        Batched tensors
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    current_length = 0
    temp_input_ids = []
    temp_ids_labels = []
    temp_attention_masks = []
    all_ids_labels = []
    temp_labels = []
    all_relation_matrices = []
    all_relation_mask = []
    temp_relation = []
    all_masked_matrice = []
    
    for item in batch:
        input_ids, attention_mask, labels, relation_matrice, ids_labels = item
        
        length = len(input_ids)

        # If adding this item would exceed max_length, process the current batch
        if current_length + length > max_length and current_length > 0:
            all_input_ids.append(torch.cat(temp_input_ids))
            all_ids_labels.append(temp_ids_labels)
            all_attention_masks.append(torch.cat(temp_attention_masks))
            all_labels.append(torch.cat(temp_labels))

            relat, relat_maks, masked_matrice = cat_matrix(temp_relation, max_length=max_length)
            all_relation_matrices.append(relat)
            all_relation_mask.append(relat_maks)
            all_masked_matrice.append(masked_matrice)
            
            # Reset temporary lists
            temp_input_ids = []
            temp_ids_labels = []
            temp_attention_masks = []
            temp_labels = []
            temp_relation = []
            current_length = 0
        
        # Add current item to temporary lists
        temp_input_ids.append(input_ids)
        temp_ids_labels.append(ids_labels)
        temp_attention_masks.append(attention_mask)
        temp_labels.append(labels)
        temp_relation.append(relation_matrice)
        current_length += length

    # Process any remaining items
    if temp_input_ids:
        all_input_ids.append(torch.cat(temp_input_ids))
        all_ids_labels.append(temp_ids_labels)
        all_attention_masks.append(torch.cat(temp_attention_masks))
        all_labels.append(torch.cat(temp_labels))

        relat, relat_maks, masked_matrice = cat_matrix(temp_relation, max_length=max_length)
        all_relation_matrices.append(relat)
        all_relation_mask.append(relat_maks)
        all_masked_matrice.append(masked_matrice)
    
    # Pad sequences
    input_ids_padded = custom_pad_sequence(all_input_ids, pad_length=max_length, batch_first=True)
    ids_labels_padded = all_ids_labels
    attention_masks_padded = custom_pad_sequence(all_attention_masks, pad_length=max_length, batch_first=True)
    labels_padded = custom_pad_sequence(all_labels, pad_length=max_length, batch_first=True)
    
    return (
        input_ids_padded, 
        attention_masks_padded, 
        labels_padded, 
        torch.stack(all_relation_matrices, dim=0), 
        torch.stack(all_relation_mask, dim=0), 
        torch.stack(all_masked_matrice, dim=0), 
        ids_labels_padded
    )


class CustomDataLoaderEvaluation(DataLoader):
    """
    Custom data loader for evaluation that includes token IDs in the batch.
    """
    def __init__(self, *args, max_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = lambda batch: custom_collate_fn_eval(batch, max_length)
        

def custom_collate_fn_eval(batch, max_length):
    """
    Custom collate function for evaluation that includes token IDs.
    
    Args:
        batch: List of (input_ids, attention_mask, labels, relation_matrix, ids_labels) tuples
        max_length: Maximum sequence length
        
    Returns:
        Batched tensors including token IDs
    """
    input_ids_padded, attention_masks_padded, labels_padded, relation_matrices, relation_mask, masked_matrice, ids_labels_padded = custom_collate_fn(batch, max_length)
    return input_ids_padded, attention_masks_padded, labels_padded, relation_matrices, relation_mask, masked_matrice, ids_labels_padded