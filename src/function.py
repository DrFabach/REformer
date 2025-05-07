
import torch
import torch.nn as nn
from tqdm import tqdm
from medkit.core.text import Entity, Span
from medkit.text.segmentation import SentenceTokenizer
from medkit.text.postprocessing import DocumentSplitter
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
from sklearn.metrics import f1_score
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import ParameterGrid

import glob
import torch
import gc

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score

def split_emea_docs_sliding_character(docs,windows=200):

    doc_splitter = DocumentSplitter_sliding_character(

        window_size=windows,
  
        attr_labels=[],  # workaround for issue 199
    )

    sentence_docs = doc_splitter.run(docs)
    return sentence_docs




class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(RelativePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rel_pos_embedding = nn.Parameter(torch.Tensor(2 * max_seq_len - 1, d_model))
        nn.init.xavier_uniform_(self.rel_pos_embedding)
    
    def forward(self, batch_size, seq_len):
        row_idx = torch.arange(seq_len).unsqueeze(1)
        col_idx = torch.arange(seq_len).unsqueeze(0)
        relative_positions = row_idx - col_idx + self.max_seq_len - 1
        relative_positions = relative_positions.unsqueeze(0).expand(batch_size, -1, -1)
        return self.rel_pos_embedding[relative_positions]
class CustomLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CustomLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes)
        )
        

    def forward(self, x):
        return self.fc(x)
        
        
class PairwiseLayer(nn.Module):
    def __init__(self, d, num_classes,hidden_dim=256):
        super(PairwiseLayer, self).__init__()
        self.custom_layer = CustomLayer(d * 2, hidden_dim, num_classes) 

    def forward(self, x, mask,position):
        batch_size, seq_len, d = x.shape

        # Initialize output tensor with the same shape as mask
        output = torch.zeros(batch_size, seq_len, seq_len,self.custom_layer.fc[-1].out_features ,device=x.device)

        # Get indices of interesting pairs



        
        mask_indices = mask.nonzero(as_tuple=True)

                # Extract corresponding pairs using the mask
        positionx = position[mask_indices[0], mask_indices[1],mask_indices[2]]
        # Extract corresponding pairs using the mask
        x_i = x[mask_indices[0], mask_indices[1]]
        x_j = x[mask_indices[0], mask_indices[2]]

        # Concatenate features of interesting pairs
        combined_features = torch.cat((x_i, x_j,positionx), dim=-1)  # Shape: (num_interesting_pairs, 2 * d)

        # Apply custom layer
        output_values = self.custom_layer(combined_features)  # Shape: (num_interesting_pairs,)

        output[mask_indices[0], mask_indices[1], mask_indices[2]] = output_values


        return output


class RelationClassifier_attention(nn.Module):
    def __init__(self, input_dim,num_entity_types,embedding_dim,num_heads=4,hidden_dim=256,max_seq_len=512,pos_dim = 10,num_classes_relation=5):
        super(RelationClassifier_attention, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, input_dim + embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim + embedding_dim)
        )
        self.num_heads=num_heads
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity_types, embedding_dim=embedding_dim)

        self.attention = nn.MultiheadAttention(input_dim + embedding_dim, num_heads,dropout=0.1)
        self.layer_norm = nn.LayerNorm(input_dim + embedding_dim)
        # Optional: You might want to add a final linear layer to project the attention output
        self.pairwise_layer = PairwiseLayer(input_dim + embedding_dim + pos_dim, hidden_dim=hidden_dim, num_classes=num_classes_relation)
        self.rel_pos_embedding = RelativePositionEmbedding(max_seq_len, pos_dim*2)
       

    def forward(self, x,entity_labels,attention_mask,sentence_mask):
        batch_size, seq_len, hidden_dim = x.shape


        entity_labels = entity_labels.long()  

        entity_embeddings = self.entity_embedding(entity_labels)

        # Transform each token embedding
        combined = torch.cat([x, entity_embeddings], dim=-1)
        transformed = self.transform(combined)
        #sentence_mask
        if sentence_mask is not None:
            # Expand the mask to (batch_size * num_heads, seq_len, seq_len)
            sentence_mask = sentence_mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            sentence_mask = sentence_mask.reshape(batch_size * self.num_heads, seq_len, seq_len)
            # Convert 0s to -inf and 1s to 0
            sentence_mask = sentence_mask.float().masked_fill(sentence_mask == 0, False).masked_fill(sentence_mask == 1,True)
       
        # Reshape for attention (seq_len, batch_size, hidden_dim)
        transformed = transformed.permute(1, 0, 2)
        
        # Apply self-attention
        attn_output, _ = self.attention(transformed, transformed, transformed,  attn_mask=sentence_mask)
        #attn_output, attn_weights = self.attention(transformed, transformed, transformed)
        attn_output = attn_output.permute(1, 0, 2)

        output =self.layer_norm(attn_output)


        
        attention_mask = attention_mask.reshape(batch_size, seq_len, seq_len).float().masked_fill(attention_mask == 0, False).masked_fill(attention_mask == 1,True).bool()
       
        rel_pos_emb = self.rel_pos_embedding(batch_size,seq_len)     


        output = self.pairwise_layer(output,attention_mask,rel_pos_emb)
        return output



# Combine BERT and relation classifier
class RelationExtractionModel(nn.Module):
    def __init__(self, bert_model, relation_classifier):
        super(RelationExtractionModel, self).__init__()
        self.bert = bert_model


        self.relation_classifier = relation_classifier

    def forward(self, input_ids, attention_mask, entity_labels, pair_mask,sentence_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state

        relation_matrix = self.relation_classifier(token_embeddings, entity_labels,pair_mask,sentence_mask)


        return relation_matrix
    
    


# Split an EMEA document into multiples "mini-docs",
# one for each sentence
def split_emea_docs(docs):
    sentence_tokenizer = SentenceTokenizer(
        output_label="sentence",
        # EMEA docs contain one sentence per line so splitting on newlines should be enough
        split_on_newlines=False,
        punct_chars=["?", "!","."],
        keep_punct=True,
    )
    doc_splitter = DocumentSplitter(
        segment_label="sentence",
        attr_labels=[],  # workaround for issue 199
    )

    for doc in docs:
        sentence_segs = sentence_tokenizer.run([doc.raw_segment])
        for sentence_seg in sentence_segs:
            doc.anns.add(sentence_seg)
    sentence_docs= []
    sentence_docs = doc_splitter.run(docs)
    return sentence_docs
def validate(model, dataloader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0
    res=[]
    with torch.no_grad():  # Disable gradient computation
        #with tqdm(dataloader, desc='Validation', unit='batch') as tqdm_dataloader:
        for batch in dataloader:#tqdm_dataloader:
            input_ids, attention_mask, labels, target_relations, pair_mask,sentence_mask = batch
            


            # Forward pass
            relation_matrix = model(input_ids, attention_mask, labels, pair_mask,sentence_mask)
            res.append(relation_matrix)
            # Calculate loss
            relation_matrix = relation_matrix.float()
            target_relations = target_relations.long()
            relation_loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])

            loss = relation_loss

            # Accumulate loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Update progress bar
            #
            
            
            
            
            #tqdm_dataloader.set_postfix({'val_loss': loss.item()})

    # Calculate average loss
    avg_loss = total_loss / total_samples

    return avg_loss,res 
   

class NERDataset(Dataset):
    def __init__(self, docs, tokenizer, label_map,commun,map_relation):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.data = []
        self.commun =commun

        for doc in docs:
            text = doc.text
            entities = [(ent.spans[0].start,ent.spans[0].end ,ent.label,ent.uid) for ent in doc.anns.get_entities()]
            relations=[(rel.source_id, rel.target_id, map_relation[rel.label])  for rel in doc.anns.get_relations()] 

            input_ids, attention_mask, labels, relations,ent_ids = self.align_labels(text, entities, relations )
            self.data.append((input_ids, attention_mask, labels,relations,ent_ids))

    def align_labels(self, text, entities,relations):
        tokenized_inputs = self.tokenizer(text, return_offsets_mapping=True, truncation=True, padding=False)
        offsets = tokenized_inputs.pop("offset_mapping")
        word_ids = tokenized_inputs.word_ids()
        
        labels = ["O"] * len(offsets)
        ent_ids = [None] * len(offsets)
        
        for start, end, label, ent_id in entities:
            ent_encours = None
            for idx, (offset_start, offset_end) in enumerate(offsets):
                word_idx = word_ids[idx]
                
                if word_idx is None:
                    continue  # Skip special tokens like [CLS] or [SEP]
                
                # Check if the token is fully or partially inside the entity
                if offset_start >= start and offset_end <= end and offset_start<offset_end+1:
                    if offset_start == start:
                        
                        if ent_encours is None :
                            labels[idx] = f"B-{label}"
                            ent_ids[idx] = ent_id
                            ent_encours=ent_id
                        else : 
                            labels[idx] = f"I-{label}" 
                    else:
                        labels[idx] = f"I-{label}"
                elif start < offset_end and end > offset_start and offset_start< offset_end+1:
                    # Handles cases where entity spans across multiple tokens
                    if offset_start == start:
                        
                        if ent_encours is None :
                            labels[idx] = f"B-{label}"
                            ent_ids[idx] = ent_id
                            ent_encours=ent_id
                        else : 
                            labels[idx] = f"I-{label}"
                    else:
                        labels[idx] = f"I-{label}"
        label_ids = [self.label_map[label] for label in labels]

        relations = set(relations)


        dicts={a+b: [] for a,b,c in relations}


        for x,y,z  in relations:
        
            dicts[x+y].append(z)
        for x in dicts.values():
            if len(x)>1:
                print(x)
       
        matrice_relat= create_relation_matrix(ent_ids,relations ,commun=self.commun)  

        return (
            torch.tensor(tokenized_inputs["input_ids"]),
            torch.tensor(tokenized_inputs["attention_mask"]),
            torch.tensor(label_ids),
            matrice_relat,
            ent_ids
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
      

import torch

def generate_common_parent_relations(relations):
    parent_to_children = {}
    for child,parent  in relations:
        parent_to_children.setdefault(parent, set()).add(child)
    
    common_parent_relations = set()
    for parent, children in parent_to_children.items():
        children_list = list(children)
        for i in range(len(children_list)):
            for j in range(i+1, len(children_list)):
                common_parent_relations.add((children_list[i], children_list[j]))
                common_parent_relations.add((children_list[j], children_list[i]))
    
    return common_parent_relations

def create_relation_matrix2(ent_ids, relations,commun=False):
    if commun : 
        relations = set(relations) | generate_common_parent_relations(relations)
    n = len(ent_ids)
    
    # Create a mask for None values
    none_mask = torch.tensor([id is None for id in ent_ids])
    
    # Create the initial matrix
    matrix = torch.zeros((n, n))
    
    # Set -100 for None values
    matrix[none_mask] = -100
    matrix[:, none_mask] = -100
    
    # Set -100 for diagonal (same entity)
    matrix.fill_diagonal_(-100)
    
    # Create a set of relations for faster lookup
    relation_set = set(relations) | set((b, a) for a, b in relations)
    
    # Create a boolean mask for relations
    relation_mask = torch.tensor([
        [
            (ent_ids[i], ent_ids[j]) in relation_set
            for j in range(n)
        ]
        for i in range(n)
    ])
   
    # Set 1 for existing relations
    matrix[relation_mask] = 1
    
    return matrix

import torch
import numpy as np
from scipy.sparse import csr_matrix
def create_relation_matrix(ent_ids, relations, commun=False):
    n = len(ent_ids)
    
    # Create a dictionary mapping entity IDs to their indices
    ent_to_idx = {eid: idx for idx, eid in enumerate(ent_ids) if eid is not None}
    
    # Initialize lists to store the sparse matrix data
    rows, cols, data = [], [], []
    seen={}
    # Process relations
    for x, y, z in relations:
    # Check if the (x, y) pair is already in the seen dictionary
        if (x, y) in relations:
        # Replace the z value with the specified replacement value
            seen[(x, y)] = (x, y, 16)
        else:
        # Add the (x, y) pair with its current z value to the seen dictionary
            seen[(x, y)] = (x, y, z)

    relations = list(seen.values())
    for e1, e2, rel_class in relations:
        if e1 in ent_to_idx and e2 in ent_to_idx:
            i, j = ent_to_idx[e1], ent_to_idx[e2]
            rows.append(i)
            cols.append(j)
            data.append(rel_class)


    # Create sparse matrix
    
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

    matrix = torch.sparse_csr_tensor(
        torch.tensor(sparse_matrix.indptr),
        torch.tensor(sparse_matrix.indices),
        torch.tensor(sparse_matrix.data),
        size=(n, n)
    ).to_dense()
    
    
    # Set -100 for diagonal and None entities
    matrix.fill_diagonal_(-100)
    none_mask = torch.tensor([id is None for id in ent_ids])
    matrix[none_mask,:] = -100
    matrix[:, none_mask] = -100

    return matrix
import random
import pandas as pd

def print_debug_samples(dataset, tokenizer, num_samples=5):
    for i in random.sample(range(len(dataset)),num_samples ) :
        input_ids, attention_mask, labels, relations,token = dataset[i]
        
        # Decode token IDs to text
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        range_interessant = [i for i,x in enumerate(token) if x is not None]
        # Map label IDs to string labels
        string_labels = [reverse_bio_label_map[label_id.item()] for label_id in labels]
        text_sel = tokenizer.decode([x for i,x in enumerate(input_ids) if string_labels[i]!="O"])
        labels = [x for x in string_labels if x!="O"]
        print(f"Sample {i + 1}:")
        print(f"Text: {text} ")
        print(f"Text ner: {text_sel} ")
        
        print(f"Labels: {labels}")
        print("=" * 50)
        print(len(labels))
        print(len(text_sel))
       


   
        text = [tokenizer.decode(input_ids[i:i+2]) for i,x in enumerate(input_ids) ]
        column_names=row_names = text


   
        df_relation = pd.DataFrame(relations.detach().numpy(), index=row_names, columns=column_names)   
        df_relation
        display(df_relation.iloc[range_interessant,range_interessant])
        #print(f"relation:\n\n{df_relation} ")
        
       
        print("=" * 50)
# Create a few samples for debugging

#print_debug_samples(dataset_eval, tokenizer, num_samples=5)


def create_pair_mask(entity_matrix):
    seq_len = entity_matrix.size(0)
    pair_mask = (entity_matrix != -100).float()

    
    return pair_mask

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, default_collate
def custom_pad_sequence(sequences, pad_length, batch_first=True, padding_value=0):
    # First, pad to the longest sequence in the batch
    padded = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
    
    # Get current sequence length
    current_length = padded.size(1) if batch_first else padded.size(0)
    
    # If current_length is less than pad_length, add more padding
    if current_length < pad_length:
        if batch_first:
            padding = torch.full((padded.size(0), pad_length - current_length, *padded.size()[2:]), 
                                 padding_value, 
                                 dtype=padded.dtype, 
                                 device=padded.device)
            padded = torch.cat([padded, padding], dim=1)
        else:
            padding = torch.full((pad_length - current_length, padded.size(1), *padded.size()[2:]), 
                                 padding_value, 
                                 dtype=padded.dtype, 
                                 device=padded.device)
            padded = torch.cat([padded, padding], dim=0)
    
    # If current_length is more than pad_length, truncate
    elif current_length > pad_length:
        if batch_first:
            padded = padded[:, :pad_length]
        else:
            padded = padded[:pad_length]
    
    return padded




from accelerate import Accelerator
from tqdm import tqdm
import torch.cuda
from seqeval.metrics import classification_report, accuracy_score  # Import accuracy_score
from transformers import  AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch
from seqeval.metrics import classification_report, accuracy_score


import torch.nn as nn


    
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
def custom_collate_fn(batch, max_length):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    current_length = 0
    temp_input_ids = []
    temp_ids_labels = []
    temp_attention_masks = []
    all_ids_labels=[]
    temp_labels = []
    all_relation_matrices=[]
    all_relation_mask = []
    temp_relation=[]
    all_masked_matrice = []
    for item in batch:
        input_ids, attention_mask, labels, relation_matrice,ids_labels = item
#        input_ids=torch.tensor(input_ids)
#        attention_mask = torch.tensor(attention_mask)
#        labels=torch.tensor(labels)
#        relation_matrice=torch.tensor(relation_matrice)

        
        length = len(input_ids)

        if current_length + length > max_length and current_length>0:

            all_input_ids.append(torch.cat(temp_input_ids))
            all_ids_labels.append(temp_ids_labels)
            all_attention_masks.append(torch.cat(temp_attention_masks))
            all_labels.append(torch.cat(temp_labels))

            relat,relat_maks,masked_matrice = cat_matrix(temp_relation,max_length=max_length)
            all_relation_matrices.append(relat)
            all_relation_mask.append(relat_maks)
            all_masked_matrice.append(masked_matrice)
            temp_input_ids = []
            temp_ids_labels = []
            temp_attention_masks = []
            temp_labels = []
            temp_relation = []
            current_length = 0
        
        temp_input_ids.append(input_ids)
        temp_ids_labels.append(ids_labels)
        temp_attention_masks.append(attention_mask)
        temp_labels.append(labels)
        temp_relation.append(relation_matrice)
        current_length += length

    if temp_input_ids:
        all_input_ids.append(torch.cat(temp_input_ids))
        all_ids_labels.append(temp_ids_labels)
        all_attention_masks.append(torch.cat(temp_attention_masks))
        all_labels.append(torch.cat(temp_labels))

        relat,relat_maks,masked_matrice = cat_matrix(temp_relation,max_length=max_length)
        all_relation_matrices.append(relat)
        all_relation_mask.append(relat_maks)
        all_masked_matrice.append(masked_matrice)
        

    input_ids_padded = custom_pad_sequence(all_input_ids,pad_length=max_length, batch_first=True)
    ids_labels_padded =all_ids_labels
    attention_masks_padded = custom_pad_sequence(all_attention_masks,pad_length=max_length, batch_first=True)
    labels_padded = custom_pad_sequence(all_labels,pad_length=max_length, batch_first=True)
    

    return input_ids_padded, attention_masks_padded, labels_padded,torch.stack(all_relation_matrices,dim=0),torch.stack(all_relation_mask,dim=0),torch.stack(all_masked_matrice,dim=0),ids_labels_padded

def custom_collate_fn_old(batch, max_length):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    current_length = 0
    temp_input_ids = []
    temp_ids_labels = []
    temp_attention_masks = []
    all_ids_labels=[]
    temp_labels = []
    all_relation_matrices=[]
    all_relation_mask = []
    temp_relation=[]
    all_masked_matrice = []
    for item in batch:
        input_ids, attention_mask, labels, relation_matrice,ids_labels = item
#        input_ids=torch.tensor(input_ids)
#        attention_mask = torch.tensor(attention_mask)
#        labels=torch.tensor(labels)
#        relation_matrice=torch.tensor(relation_matrice)

        
        length = len(input_ids)

        if current_length + length > max_length and current_length>0:

            all_input_ids.append(torch.cat(temp_input_ids))
            all_ids_labels.append(torch.cat(temp_ids_labels))
            all_attention_masks.append(torch.cat(temp_attention_masks))
            all_labels.append(torch.cat(temp_labels))

            relat,relat_maks,masked_matrice = cat_matrix(temp_relation,max_length=max_length)
            all_relation_matrices.append(relat)
            all_relation_mask.append(relat_maks)
            all_masked_matrice.append(masked_matrice)
            temp_input_ids = []
            temp_ids_labels = []
            temp_attention_masks = []
            temp_labels = []
            temp_relation = []
            current_length = 0
        
        temp_input_ids.append(input_ids)
        temp_ids_labels.append(ids_labels)
        temp_attention_masks.append(attention_mask)
        temp_labels.append(labels)
        temp_relation.append(relation_matrice)
        current_length += length

    if temp_input_ids:
        all_input_ids.append(torch.cat(temp_input_ids))
        all_ids_labels.append(torch.cat(temp_ids_labels))
        all_attention_masks.append(torch.cat(temp_attention_masks))
        all_labels.append(torch.cat(temp_labels))

        relat,relat_maks,masked_matrice = cat_matrix(temp_relation,max_length=max_length)
        all_relation_matrices.append(relat)
        all_relation_mask.append(relat_maks)
        all_masked_matrice.append(masked_matrice)
        

    input_ids_padded = custom_pad_sequence(all_input_ids,pad_length=max_length, batch_first=True)
    ids_labels_padded =custom_pad_sequence(all_ids_labels,pad_length=max_length, batch_first=True)
    attention_masks_padded = custom_pad_sequence(all_attention_masks,pad_length=max_length, batch_first=True)
    labels_padded = custom_pad_sequence(all_labels,pad_length=max_length, batch_first=True)
    

    
    


    #print([(i.shape) for i in (all_relation_matrices)])
    return input_ids_padded, attention_masks_padded, labels_padded,
    torch.stack(all_relation_matrices,dim=0),torch.stack(all_relation_mask,dim=0),
    torch.stack(all_masked_matrice,dim=0),
    ids_labels_padded
from torch.nn.utils.rnn import pad_sequence
class CustomDataLoader(DataLoader):
    def __init__(self, *args, max_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = lambda batch: custom_collate_fn(batch, max_length)
        
  #model à partir du 29 aout.      
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, default_collate


        
def cat_matrix(all_relation_matrices,max_length):

    final_matrix = torch.full((max_length, max_length), -100)
    final_matrix2 = torch.full((max_length, max_length), 0)
    current_idx = 0
    for matrix in all_relation_matrices:
        
     
        size = matrix.size(0)
        
        if current_idx+size <= max_length:
            final_matrix[current_idx:current_idx+size, current_idx:current_idx+size] = matrix
            final_matrix2[current_idx:current_idx+size, current_idx:current_idx+size]=1

        else:
            final_matrix[current_idx:max_length, current_idx:max_length]=matrix[:max_length-current_idx, :max_length-current_idx ]
            final_matrix2[current_idx:max_length, current_idx:max_length]=1

        current_idx += size
           
    pair_masks_padded= create_pair_mask(final_matrix)

    return(final_matrix,pair_masks_padded,final_matrix2)











from medkit.core import Attribute, Operation
from medkit.core.text import (
    Entity,
    ModifiedSpan,
    Relation,
    Segment,
    Span,
    TextAnnotation,
    TextDocument,
    span_utils,
)

from medkit.text.postprocessing.alignment_utils import compute_nested_segments
import copy



class DocumentSplitter2(Operation):
    """Split text documents using its segments as a reference.

    The resulting 'mini-documents' contain the entities belonging to each
    segment along with their attributes.

    This operation can be used to create datasets from medkit text documents.
    """

    def __init__(
        self,
        
        segment_label: str,
        entity_labels= None,
        attr_labels = None,
        relation_labels = None,
        name = None,
        uid= None,
        label_center = None
    ):
        """Instantiate the document splitter

        Parameters
        ----------
        segment_label : str
            Label of the segments to use as references for the splitter
        entity_labels : list of str, optional
            Labels of entities to be included in the mini documents.
            If None, all entities from the document will be included.
        attr_labels : list of str, optional
            Labels of the attributes to be included into the new annotations.
            If None, all attributes will be included.
        relation_labels : list of str, optional
            Labels of relations to be included in the mini documents.
            If None, all relations will be included.
        name : str, optional
            Name describing the splitter (default to the class name).
        uid : str, Optional
            Identifier of the operation
        """
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.segment_label = segment_label
        self.entity_labels = entity_labels
        self.attr_labels = attr_labels
        self.relation_labels = relation_labels
        self.label_center=label_center



    def run(self, docs: list[TextDocument]) -> list[TextDocument]:
        """Split docs into mini documents

        Parameters
        ----------
        docs: list of TextDocument
            List of text documents to split

        Returns
        -------
        list of TextDocument
            List of documents created from the selected segments
        """
        segment_docs = []

        for doc in docs:
            segments = doc.anns.get_segments(label=self.segment_label)

            # filter entities
            entities = (
                doc.anns.get_entities()
                if self.entity_labels is None
                else [ent for label in self.entity_labels for ent in doc.anns.get_entities(label=label)]
            )

            # align segment and entities (fully contained)
            if self.label_center is None :
            
                segment_and_entities = compute_nested_segments(segments, entities)
            else : 
                index_entities_center = [i for i,x in enumerate(entities) if x.label in self.label_center ]

                
                segment_and_entities=[]
                for i in range(len(segments)):


                    segment_and_entities.append(compute_nested_segments([segments[i]],  [x for j,x in enumerate(entities) ])[0])
            
            # filter relations in the document
            relations = (
                doc.anns.get_relations()
                if self.relation_labels is None
                else [rel for label in self.relation_labels for rel in doc.anns.get_relations(label=label)]
            )

            # Iterate over all segments and corresponding nested entities
            for segment, nested_entities in segment_and_entities:
                # filter relations in nested entities
                entities_uid = {ent.uid for ent in nested_entities}
                nested_relations = [
                    relation
                    for relation in relations
                    if relation.source_id in entities_uid and relation.target_id in entities_uid
                ]
                # create new document from segment
                segment_doc = self._create_segment_doc(
                    segment=segment,
                    entities=nested_entities,
                    relations=nested_relations,
                    doc_source=doc,
                )
                segment_docs.append(segment_doc)

        return segment_docs


    def _create_segment_doc(
        self,
        segment: Segment,
        entities: list[Entity],
        relations: list[Relation],
        doc_source: TextDocument,
    ) -> TextDocument:
        """Create a TextDocument from a segment and its entities.
        The original zone of the segment becomes the text of the document.

        Parameters
        ----------
        segment : Segment
            Segment to use as reference for the new document
        entities : list of Entity
            Entities inside the segment
        relations : list of Relation
            Relations inside the segment
        doc_source : TextDocument
            Initial document from which annotations where extracted

        Returns
        -------
        TextDocument
            A new document with entities, the metadata includes the original span and metadata
        """
        normalized_spans = span_utils.normalize_spans(segment.spans)

        # create an empty mini-doc with the raw text of the segment
        offset, end_span = normalized_spans[0].start, normalized_spans[-1].end
        metadata = doc_source.metadata.copy()
        metadata.update(segment.metadata)

        segment_doc = TextDocument(text=doc_source.text[offset:end_span], metadata=metadata)

        # handle provenance
        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(segment_doc, self.description, source_data_items=[segment])

        # Copy segment attributes
        segment_attrs = self._filter_attrs_from_ann(segment)
        for attr in segment_attrs:
            new_doc_attr = attr.copy()
            segment_doc.attrs.add(new_doc_attr)
            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(
                    new_doc_attr,
                    self.description,
                    source_data_items=[attr],
                )

        # Add selected entities
        uid_mapping = {}
        for ent in entities:
            spans = []
            for span in ent.spans:
                # relocate entity spans using segment offset
                if isinstance(span, Span):
                    spans.append(Span(span.start - offset, span.end - offset))
                else:
                    replaced_spans = [Span(sp.start - offset, sp.end - offset) for sp in span.replaced_spans]
                    spans.append(ModifiedSpan(length=span.length, replaced_spans=replaced_spans))
            # define the new entity
            relocated_ent = Entity(
                text=ent.text,
                label=ent.label,
                spans=spans,
                metadata=ent.metadata.copy(),
            )
            # add mapping for relations
            uid_mapping[ent.uid] = relocated_ent.uid

            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(relocated_ent, self.description, source_data_items=[ent])

            # Copy entity attributes
            entity_attrs = self._filter_attrs_from_ann(ent)
            for attr in entity_attrs:
                new_ent_attr = attr.copy()
                relocated_ent.attrs.add(new_ent_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        new_ent_attr,
                        self.description,
                        source_data_items=[attr],
                    )

            # add entity to the new document
            segment_doc.anns.add(relocated_ent)

        for rel in relations:
            relation = Relation(
                label=rel.label,
                source_id=uid_mapping[rel.source_id],
                target_id=uid_mapping[rel.target_id],
                metadata=rel.metadata.copy(),
            )
            # handle provenance
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(relation, self.description, source_data_items=[rel])

            # Copy relation attributes
            relation_attrs = self._filter_attrs_from_ann(rel)
            for attr in relation_attrs:
                new_rel_attr = attr.copy()
                relation.attrs.add(new_rel_attr)
                # handle provenance
                if self._prov_tracer is not None:
                    self._prov_tracer.add_prov(
                        new_rel_attr,
                        self.description,
                        source_data_items=[attr],
                    )

            # add relation to the new document
            segment_doc.anns.add(relation)

        return segment_doc

    def _filter_attrs_from_ann(self, ann: TextAnnotation) -> list[Attribute]:
        """Filter attributes from an annotation using 'attr_labels'"""
        return (
            ann.attrs.get()
            if self.attr_labels is None
            else [attr for label in self.attr_labels for attr in ann.attrs.get(label=label)]
        )




from typing import Union, List, Optional
from medkit.core import Attribute, Operation
from medkit.core.text import (
    Entity,
    ModifiedSpan,
    Relation,
    Segment,
    Span,
    TextAnnotation,
    TextDocument,
    span_utils,
)

from medkit.text.postprocessing.alignment_utils import compute_nested_segments

class DocumentSplitter_sliding_window(Operation):
    """Split text documents into sliding windows of segments.

    Parameters
    ----------
    segment_label : str
        Label of the segments to use as references for the splitter
    window_size : int
        Number of segments to include in each window
    stride : int
        Number of segments to slide the window by
    entity_labels : list[str] | None, optional
        Labels of entities to be included in the mini documents.
        If None, all entities from the document will be included.
    attr_labels : list[str] | None, optional
        Labels of the attributes to be included into the new annotations.
        If None, all attributes will be included.
    relation_labels : list[str] | None, optional
        Labels of relations to be included in the mini documents.
        If None, all relations will be included.
    name : str | None, optional
        Name describing the splitter (default to the class name).
    uid : str | None, optional
        Identifier of the operation
    """

    def __init__(
       self,
        segment_label: str,
        window_size: int,
        stride: int,
        entity_labels: Optional[List[str]] = None,
        attr_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.segment_label = segment_label
        self.window_size = window_size
        self.stride = stride
        self.entity_labels = entity_labels
        self.attr_labels = attr_labels
        self.relation_labels = relation_labels

    def run(self, docs: list[TextDocument]) -> list[TextDocument]:
        """Split docs into mini documents using sliding windows of segments.

        Parameters
        ----------
        docs: list of TextDocument
            List of text documents to split

        Returns
        -------
        list of TextDocument
            List of documents created from the sliding windows of segments
        """
        window_docs = []

        for doc in docs:
            segments = doc.anns.get_segments(label=self.segment_label)
            segments.sort(key=lambda s: s.spans[0].start)

            for i in range(0, len(segments), self.stride):
                window_segments = segments[i:i+self.window_size]
                if not window_segments:
                    print("azdzad")
                    break

                window_start = window_segments[0].spans[0].start
                window_end = window_segments[-1].spans[-1].end

                # Filter entities in the window
                window_entities = self._get_entities_in_window(doc, window_start, window_end)

                # Filter relations in the window
                window_relations = self._get_relations_in_window(doc, window_entities)

                # Create new document from window
                window_doc = self._create_window_doc(
                    window_start=window_start,
                    window_end=window_end,
                    segments=window_segments,
                    entities=window_entities,
                    relations=window_relations,
                    doc_source=doc,
                )
                window_docs.append(window_doc)
        
        return window_docs

    def _get_entities_in_window(self, doc: TextDocument, window_start: int, window_end: int) -> list[Entity]:
        entities = (
            doc.anns.get_entities()
            if self.entity_labels is None
            else [ent for label in self.entity_labels for ent in doc.anns.get_entities(label=label)]
        )
        return [ent for ent in entities if self._entity_in_window(ent, window_start, window_end)]

    def _get_relations_in_window(self, doc: TextDocument, window_entities: list[Entity]) -> list[Relation]:
        relations = (
            doc.anns.get_relations()
            if self.relation_labels is None
            else [rel for label in self.relation_labels for rel in doc.anns.get_relations(label=label)]
        )
        entities_uid = {ent.uid for ent in window_entities}
        return [
            relation
            for relation in relations
            if relation.source_id in entities_uid and relation.target_id in entities_uid
        ]

    def _create_window_doc(
        self,
        window_start: int,
        window_end: int,
        segments: list[Segment],
        entities: list[Entity],
        relations: list[Relation],
        doc_source: TextDocument,
    ) -> TextDocument:
        """Create a TextDocument from a window of segments and its entities."""
        metadata = doc_source.metadata.copy()
        metadata.update({
            "window_start": window_start,
            "window_end": window_end,
            "segment_indices": f"{segments[0].metadata.get('index', 'N/A')}-{segments[-1].metadata.get('index', 'N/A')}"
        })

        window_doc = TextDocument(text=doc_source.text[window_start:window_end], metadata=metadata)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(window_doc, self.description, source_data_items=[doc_source] + segments)

        uid_mapping = {}
        
        # Add segments to the new document
        for seg in segments:
            relocated_seg = self._relocate_segment(seg, window_start)
            window_doc.anns.add(relocated_seg)

        # Add entities to the new document
        for ent in entities:
            relocated_ent = self._relocate_entity(ent, window_start)
            uid_mapping[ent.uid] = relocated_ent.uid
            window_doc.anns.add(relocated_ent)

        # Add relations to the new document
        for rel in relations:
            relocated_rel = self._relocate_relation(rel, uid_mapping)
            window_doc.anns.add(relocated_rel)

        return window_doc

    def _relocate_segment(self, segment: Segment, window_start: int) -> Segment:
        spans = [Span(span.start - window_start, span.end - window_start) for span in segment.spans]
        relocated_seg = Segment(
            label=segment.label,
            spans=spans,
            text=segment.text,
            metadata=segment.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_seg, self.description, source_data_items=[segment])

        for attr in self._filter_attrs_from_ann(segment):
            new_attr = attr.copy()
            relocated_seg.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_seg

    def _relocate_entity(self, entity: Entity, window_start: int) -> Entity:
        spans = []
        for span in entity.spans:
            if isinstance(span, Span):
                spans.append(Span(span.start - window_start, span.end - window_start))
            else:
                replaced_spans = [Span(sp.start - window_start, sp.end - window_start) for sp in span.replaced_spans]
                spans.append(ModifiedSpan(length=span.length, replaced_spans=replaced_spans))

        relocated_ent = Entity(
            text=entity.text,
            label=entity.label,
            spans=spans,
            metadata=entity.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_ent, self.description, source_data_items=[entity])

        for attr in self._filter_attrs_from_ann(entity):
            new_attr = attr.copy()
            relocated_ent.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_ent

    def _relocate_relation(self, relation: Relation, uid_mapping: dict) -> Relation:
        relocated_rel = Relation(
            label=relation.label,
            source_id=uid_mapping[relation.source_id],
            target_id=uid_mapping[relation.target_id],
            metadata=relation.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_rel, self.description, source_data_items=[relation])

        for attr in self._filter_attrs_from_ann(relation):
            new_attr = attr.copy()
            relocated_rel.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_rel

    def _filter_attrs_from_ann(self, ann: TextAnnotation) -> list[Attribute]:
        """Filter attributes from an annotation using 'attr_labels'."""
        return (
            ann.attrs.get()
            if self.attr_labels is None
            else [attr for label in self.attr_labels for attr in ann.attrs.get(label=label)]
        )

    def _entity_in_window(self, entity: Entity, window_start: int, window_end: int) -> bool:
        """Check if an entity is fully contained within a window."""
        if len(entity.spans)==1:
            entity_start = entity.spans[0].start
            entity_end = entity.spans[0].end
        else :
            entity_start = doc_new[i].anns.get_entities()[-1].spans[0].replaced_spans[0].start
            entity_end = doc_new[i].anns.get_entities()[-1].spans[0].replaced_spans[0].end           
        return entity_start >= window_start and entity_end <= window_end
        
from medkit.core.text.span_utils import normalize_spans        
class DocumentSplitter_sliding_character(Operation):

    """Split text documents into sliding windows of characters with two overlapping windows.

    Parameters
    ----------
    window_size : int
        Number of characters to include in each window
    entity_labels : list[str] | None, optional
        Labels of entities to be included in the mini documents.
        If None, all entities from the document will be included.
    attr_labels : list[str] | None, optional
        Labels of the attributes to be included into the new annotations.
        If None, all attributes will be included.
    relation_labels : list[str] | None, optional
        Labels of relations to be included in the mini documents.
        If None, all relations will be included.
    name : str | None, optional
        Name describing the splitter (default to the class name).
    uid : str | None, optional
        Identifier of the operation
    """

    def __init__(
        self,
        window_size: int,
        entity_labels: Optional[List[str]] = None,
        attr_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
        overlap = True
    ):
        # Pass all arguments to super (remove self)
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.window_size = window_size
        if overlap : 
            self.stride = window_size // 2  # Set stride to half the window size
        else : 
            self.stride = window_size-20
        self.entity_labels = entity_labels
        self.attr_labels = attr_labels
        self.relation_labels = relation_labels

    def run(self, docs: list[TextDocument]) -> list[TextDocument]:
        """Split docs into mini documents using sliding windows of characters with two overlapping windows.

        Parameters
        ----------
        docs: list of TextDocument
            List of text documents to split

        Returns
        -------
        list of TextDocument
            List of documents created from the sliding windows of characters
        """
        window_docs = []

        for doc in docs:
            text = doc.text
            text_length = len(text)

            for window_start in range(0, text_length, self.stride):
                window_end = min(window_start + self.window_size, text_length)
                if window_start >= text_length:
                    break

                # Filter entities in the window
                window_entities = self._get_entities_in_window(doc, window_start, window_end)

                # Filter relations in the window
                window_relations = self._get_relations_in_window(doc, window_entities)

                # Create new document from window
                window_doc = self._create_window_doc(
                    window_start=window_start,
                    window_end=window_end,
                    entities=window_entities,
                    relations=window_relations,
                    doc_source=doc,
                )
                window_docs.append(window_doc)
        
        return window_docs

    def _get_entities_in_window(self, doc: TextDocument, window_start: int, window_end: int) -> list[Entity]:
        entities = (
            doc.anns.get_entities()
            if self.entity_labels is None
            else [ent for label in self.entity_labels for ent in doc.anns.get_entities(label=label)]
        )
        return [ent for ent in entities if self._entity_in_window(ent, window_start, window_end)]

    def _get_relations_in_window(self, doc: TextDocument, window_entities: list[Entity]) -> list[Relation]:
        relations = (
            doc.anns.get_relations()
            if self.relation_labels is None
            else [rel for label in self.relation_labels for rel in doc.anns.get_relations(label=label)]
        )
        entities_uid = {ent.uid for ent in window_entities}
        return [
            relation
            for relation in relations
            if relation.source_id in entities_uid and relation.target_id in entities_uid
        ]

    def _create_window_doc(
        self,
        window_start: int,
        window_end: int,
        entities: list[Entity],
        relations: list[Relation],
        doc_source: TextDocument,
    ) -> TextDocument:
        """Create a TextDocument from a window of characters and its entities."""
        metadata = doc_source.metadata.copy()
        metadata.update({
            "window_start": window_start,
            "window_end": window_end,
        })

        window_doc = TextDocument(text=doc_source.text[window_start:window_end], metadata=metadata)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(window_doc, self.description, source_data_items=[doc_source])

        uid_mapping = {}
        
        # Add entities to the new document
        for ent in entities:
            relocated_ent = self._relocate_entity(ent, window_start)
            uid_mapping[ent.uid] = relocated_ent.uid
            window_doc.anns.add(relocated_ent)

        # Add relations to the new document
        for rel in relations:
            relocated_rel = self._relocate_relation(rel, uid_mapping)
            window_doc.anns.add(relocated_rel)

        return window_doc

    def _relocate_entity(self, entity: Entity, window_start: int) -> Entity:
        spans = []
        for span in entity.spans:
            if isinstance(span, Span):
                spans.append(Span(span.start - window_start, span.end - window_start))
            else:
                replaced_spans = [Span(sp.start - window_start, sp.end - window_start) for sp in span.replaced_spans]
                spans.append(ModifiedSpan(length=span.length, replaced_spans=replaced_spans))

        relocated_ent = Entity(
            text=entity.text,
            label=entity.label,
            spans=spans,
            metadata=entity.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_ent, self.description, source_data_items=[entity])

        for attr in self._filter_attrs_from_ann(entity):
            new_attr = attr.copy()
            relocated_ent.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_ent

    def _relocate_relation(self, relation: Relation, uid_mapping: dict) -> Relation:
        relocated_rel = Relation(
            label=relation.label,
            source_id=uid_mapping[relation.source_id],
            target_id=uid_mapping[relation.target_id],
            metadata=relation.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_rel, self.description, source_data_items=[relation])

        for attr in self._filter_attrs_from_ann(relation):
            new_attr = attr.copy()
            relocated_rel.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_rel

    def _filter_attrs_from_ann(self, ann: TextAnnotation) -> list[Attribute]:
        """Filter attributes from an annotation using 'attr_labels'."""
        return (
            ann.attrs.get()
            if self.attr_labels is None
            else [attr for label in self.attr_labels for attr in ann.attrs.get(label=label)]
        )

    def _entity_in_window(self, entity: Entity, window_start: int, window_end: int) -> bool:
        """Check if an entity is fully contained within a window."""
        
        span_i= normalize_spans(entity.spans)
        entity_start = span_i[0].start
        entity_end = span_i[0].end
 
        return entity_start >= window_start and entity_end <= window_end
        
        
class RelationClassifier_attention_old(nn.Module):
    def __init__(self, input_dim, num_entity_types, embedding_dim, num_heads=8, max_seq_len=512):
        super(RelationClassifier_attention_old, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, input_dim + embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim + embedding_dim)
        )
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity_types, embedding_dim=embedding_dim)
        self.attention = nn.MultiheadAttention(input_dim + embedding_dim, num_heads, dropout=0.1)
        self.final_linear = nn.Linear(input_dim + embedding_dim, input_dim + embedding_dim)
        
        self.rel_pos_embedding = RelativePositionEmbedding(max_seq_len,embedding_dim)
        self.num_heads = num_heads

    def forward(self, x, entity_labels, attention_mask, sentence_mask):
        batch_size, seq_len, hidden_dim = x.shape
        entity_labels = entity_labels.long()
        entity_embeddings = self.entity_embedding(entity_labels)
        
        # Transform each token embedding
        combined = torch.cat([x, entity_embeddings], dim=-1)
        transformed = self.transform(combined)
        
        # Generate relative position embeddings
        rel_pos_emb = self.rel_pos_embedding(seq_len)
        
        if sentence_mask is not None:
            sentence_mask = sentence_mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            sentence_mask = sentence_mask.reshape(batch_size * self.num_heads, seq_len, seq_len)
            sentence_mask = sentence_mask.float().masked_fill(sentence_mask == 0, False).masked_fill(sentence_mask == 1,True)#.float().masked_fill(sentence_mask == 0, float('-inf')).masked_fill(sentence_mask == 1, 0.0)
        
        # Reshape for attention (seq_len, batch_size, hidden_dim)
        transformed = transformed.permute(1, 0, 2)
        
        # Apply self-attention with relative positional embeddings
        attn_output, _ = self.attention(transformed, transformed, transformed, 
                                        attn_mask=sentence_mask, 
                                       
                                        need_weights=False)
        
        # Add relative positional embeddings to the attention output
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
       
        # Apply final linear layer
        
        #output =self.layer_norm(attn_output)
        # Compute pairwise similarities
        output = torch.matmul(attn_output, attn_output.transpose(-1, -2))
        print(rel_pos_emb.size())
        output = output + rel_pos_emb
  
        return output
        
import numpy as np

def align_res(res, dataset,tokenizer):
    input_ids, attention_mask, labels, target_relations,ids = dataset
    res = inverse_transform_prediction(res)

    

    range_text = [i for i,x in enumerate(input_ids) if x !=0]
    range_interessant = [i for i,x in enumerate(ids) if x is not None]
     # Decode token IDs to text
    text = tokenizer.decode(input_ids[range_interessant], skip_special_tokens=False)
    
    # Map label IDs to string labels
    string_labels = [reverse_label_map[label_id.item()] for label_id in labels[range_interessant]]
    text =[tokenizer.decode(input_ids[range_interessant[i]:range_interessant[i]+3]) for i,x in enumerate(input_ids[range_interessant]) ]
    column_names=row_names = text
    text2 =   [x for i,x in enumerate(ids) if x is not None]
    df_relation = pd.DataFrame(res[np.ix_([0],range_interessant,range_interessant)][0].detach().numpy(), index=row_names, columns=column_names)   
    df_mask = pd.DataFrame(res[np.ix_([0],range_interessant,range_interessant)][0].detach().numpy(), index=text2, columns=text2)   
    #print(f"Text: { tokenizer.decode(input_ids[range_text], skip_special_tokens=False)} ")
    
    #print(f"relation:\n\n ")
    #display(HTML(df_relation.to_html()))
    #print(f"Mask:\n\n ")
    #display(HTML(df_mask.to_html()))
    result = []
    for col in df_mask.columns:
        for idx in df_mask.index:
            if df_mask.loc[idx, col] == 1.0:
                result.append((col, idx))


    return(result)
    
def inverse_transform_prediction(logits, threshold=0.5):
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)
    
    # Apply threshold to get binary predictions
    predictions = (probabilities > threshold).float()
    
    return predictions 
 
# Create a few samples for debugging
#tokenizer = CamembertTokenizerFast.from_pretrained("roberta-base")
def calculate_f1_score(y_true, y_pred, average='macro', exclude_classes=[0]):
    """
    Calculate the F1 score, precision, and recall for multi-class classification,
    excluding specified classes.
    
    Args:
        y_true (list or numpy array): Ground truth (correct) target values.
        y_pred (list or numpy array): Estimated targets as returned by a classifier.
        average (str): Type of averaging performed on the data.
                       'macro' calculates metrics for each class and averages them.
                       'micro' calculates metrics globally by counting the total true positives, 
                       false negatives, and false positives.
                       'weighted' takes into account the number of instances for each class.
        exclude_classes (list): List of class labels to exclude from the calculation.
    
    Returns:
        f1 (float): F1 score.
        precision (float): Precision score.
        recall (float): Recall score.
    """
    # Convert inputs to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create a mask to filter out the excluded classes
    mask = np.isin(y_true, exclude_classes, invert=True)
    
    # Apply the mask to both y_true and y_pred
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate metrics using the filtered data
    f1 = f1_score(y_true_filtered, y_pred_filtered, average=average)
    precision = precision_score(y_true_filtered, y_pred_filtered, average=average)
    recall = recall_score(y_true_filtered, y_pred_filtered, average=average)
    
    return f1, precision, recall


def pad_with_nan(array_list):
    max_length = max(len(arr) for arr in array_list)
    padded_list = []
    for arr in array_list:
        padding_length = max_length - len(arr)
        padded_arr = np.pad(arr, (0, padding_length), constant_values=np.nan)
        padded_list.append(padded_arr)
    return np.array(padded_list)

# Pad the arrays
def train_with_cross_validation(dataset_sentences,embedding_dim,num_heads,hidden_dim,pos_dim, writer ,num_entity_types=21,num_folds=5, num_epochs=20, batch_size=4, lr=1e-5, max_length=512,name_bert_model="almanach/camembert-bio-base",):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    temp_bert_model = AutoModel.from_pretrained(name_bert_model)
    bert_hidden_size = temp_bert_model.config.hidden_size
    if (bert_hidden_size + int(embedding_dim)) % int(num_heads) != 0:
        return np.inf  # Return a high value to indicate an invalid combination
    all_train_losses = []
    all_val_losses = []
    all_val_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_sentences)):
       
        bert_model = AutoModel.from_pretrained(name_bert_model)#'dmis-lab/biobert-base-cased-v1.2')
        bert_model.resize_token_embeddings(len(tokenizer))
        relation_classifier=RelationClassifier_attention(bert_model.config.hidden_size,num_entity_types=num_entity_types,
                                                         embedding_dim=embedding_dim,num_heads=num_heads,hidden_dim=hidden_dim,
                                     pos_dim = pos_dim,num_classes_relation=17)
        print(f"Fold {fold + 1}/{num_folds}")

        # Create data loaders for this fold
        train_dataset = Subset(dataset_sentences, train_idx)
        val_dataset = Subset(dataset_sentences, val_idx)
        
        train_dataloader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True, max_length=max_length)
        val_dataloader = CustomDataLoader(val_dataset, batch_size=batch_size, shuffle=False, max_length=max_length)

        # Create a new instance of the model for each fold
        model = RelationExtractionModel(bert_model, relation_classifier)

        # Setup training
        
        loss_fn =  nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        accelerator = Accelerator()
        model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader
        )

        # Initialize TensorBoard writer for this fold
        

        train_losses = []
        val_losses = []
        val_f1_scores = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, _ = batch
                
                relation_matrix = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
                relation_matrix = relation_matrix.float()
                target_relations = target_relations.long()
                relation_loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])
           
                loss = relation_loss
                total_loss += loss.item()
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)


            # Validation
            model.eval()
            val_loss, val_f1, _ = validate(model, val_dataloader, loss_fn)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)


            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
            if (epoch ==3 and val_loss>1) or(epoch ==6 and val_loss>0.8) or (epoch ==1 and avg_train_loss>1.7) :
                
                break    
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_f1_scores.append(val_f1_scores)

        # Clean up GPU memory after each fold
        del model, optimizer, scheduler, train_dataloader, val_dataloader, loss_fn
        torch.cuda.empty_cache()
        gc.collect()


    all_train_losses_padded = pad_with_nan(all_train_losses)
    all_val_losses_padded = pad_with_nan(all_val_losses)
    all_val_f1_scores_padded = pad_with_nan(all_val_f1_scores)

    # Calculate means, ignoring NaN values
    avg_train_loss = np.nanmean(all_train_losses_padded, axis=0)
    avg_val_loss = np.nanmean(all_val_losses_padded, axis=0)
    avg_val_f1 = np.nanmean(all_val_f1_scores_padded, axis=0)


    print("\nAverage performance across folds:")
    for epoch in range(len(avg_train_loss)):
        print(f'Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss[epoch]:.4f}, '
              f'Avg Val Loss: {avg_val_loss[epoch]:.4f}, Avg Val F1: {avg_val_f1[epoch]:.4f}')
        writer.add_scalar('Loss/Train', avg_train_loss[epoch], epoch+1)
        writer.add_scalar('Loss/validation', avg_val_loss[epoch], epoch+1)
        writer.add_scalar('F1/validation', avg_val_f1[epoch], epoch+1)

    return all_train_losses, all_val_losses, all_val_f1_scores
def validate(model, dataloader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_samples = 0
    res = []
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, ids_token = batch

            # Forward pass
            relation_matrix = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
            res.append(relation_matrix)
            relation_matrix = relation_matrix.float()
            target_relations = target_relations.long()
            # Calculate loss (assuming target_relations contain class labels)
            loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])

            # Accumulate loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Get the predicted class by finding the index with the highest score for each sample
            preds = torch.argmax(relation_matrix[pair_mask == 1], dim=-1)
            targets = target_relations[pair_mask == 1]

            # Accumulate predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / total_samples

    # Calculate F1 score (use average='macro' for multi-class)

    f1, precision, recall = calculate_f1_score(all_targets, all_preds, average='micro')

    return avg_loss, f1, res

import torch
import torch.nn as nn

class JointNERREModel(nn.Module):
    def __init__(self, bert_model, num_entity_types=21, embedding_dim=768, num_heads=4, 
                 hidden_dim=256, max_seq_len=512, pos_dim=10, num_classes_relation=15):
        super(JointNERREModel, self).__init__()
        self.bert = bert_model
        bert_hidden_size = bert_model.config.hidden_size
        
        # NER components
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_classifier = nn.Linear(bert_hidden_size, num_entity_types)
        
        # RE components
        self.relation_classifier = RelationClassifier_attention(
            input_dim=bert_hidden_size,
            num_entity_types=num_entity_types,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            pos_dim=pos_dim,
            num_classes_relation=num_classes_relation
        )

    def forward(self, input_ids, attention_mask, entity_labels=None, pair_mask=None, sentence_mask=None):
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state
        
        # NER prediction
        ner_output = self.ner_dropout(token_embeddings)
        ner_logits = self.ner_classifier(ner_output)
        
        # If we're training, use gold entity labels for RE
        # If we're inferring, use predicted entity labels
        if entity_labels is None:
            entity_labels = torch.argmax(ner_logits, dim=-1)
        
        # RE prediction
        relation_matrix = self.relation_classifier(
            token_embeddings, 
            entity_labels,
            pair_mask,
            sentence_mask
        )
        
        return {
            'ner_logits': ner_logits,
            'relation_matrix': relation_matrix
        }

class JointLoss(nn.Module):
    def __init__(self, ner_weight=1.0, re_weight=1.0):
        super(JointLoss, self).__init__()
        self.ner_criterion = nn.CrossEntropyLoss()
        self.re_criterion = nn.CrossEntropyLoss()
        self.ner_weight = ner_weight
        self.re_weight = re_weight
    
    def forward(self, outputs, ner_labels, re_labels, attention_mask=None, pair_mask=None):
        # NER loss
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs['ner_logits'].view(-1, outputs['ner_logits'].shape[-1])
        active_labels = ner_labels.view(-1)
        active_logits = active_logits[active_loss]
        active_labels = active_labels[active_loss]
        ner_loss = self.ner_criterion(active_logits, active_labels)
        
        # RE loss
        re_loss = 0
        if pair_mask is not None:
            active_pairs = pair_mask.bool()
            active_re_logits = outputs['relation_matrix'][active_pairs]
            active_re_labels = re_labels[active_pairs]
            re_loss = self.re_criterion(active_re_logits, active_re_labels)
        
        # Combined loss
        total_loss = self.ner_weight * ner_loss + self.re_weight * re_loss
        
        return {
            'total_loss': total_loss,
            'ner_loss': ner_loss,
            're_loss': re_loss
        }

