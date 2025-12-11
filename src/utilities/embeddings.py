import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = 'allenai/specter2_base'
DEFAULT_SPARSE_MODEL = 'naver/splade-cocondenser-ensembledistil'

class EmbeddingGenerator:
    def __init__(self, dense_model: str = DEFAULT_DENSE_MODEL, sparse_model: str = DEFAULT_SPARSE_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        logger.info(f"Using device: {self.device}")

        # load dense model (SPECTER2)
        self.dense_model_name = dense_model
        self.dense_model = SentenceTransformer(self.dense_model_name).to(self.device)
        logger.info(f"Loaded Dense Model: {self.dense_model_name}")

        # load sparse model (SPLADE)
        self.sparse_model_id = sparse_model
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_id)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_id).to(self.device)
        logger.info(f"Loaded Sparse Model: {self.sparse_model_id}")

    def generate_dense_embeddings(self, texts):
        return self.dense_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def generate_sparse_embedding(self, text):
        tokens = self.sparse_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.sparse_model(**tokens)
        # SPLADE aggregation (max pooling over log(1 + relu(logits)))
        logits = output.logits
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits)) * tokens.attention_mask.unsqueeze(-1), 
            dim=1
        )
        # Convert to dict ~ {token_id: weight}
        vector = values.squeeze().cpu().numpy()
        non_zero_indices = vector.nonzero()[0]
        return {int(idx): float(vector[idx]) for idx in non_zero_indices}

    def generate_embeddings(self, paper_df: pd.DataFrame) -> pd.DataFrame:
        df = paper_df.copy()
        
        sep_token = self.dense_model.tokenizer.sep_token
        df['specter_input'] = df['title'] + sep_token + df['abstract']
        df['dense_vector'] = list(
            self.generate_dense_embeddings(df['specter_input'].tolist()).cpu().numpy()
        )
        df['sparse_vector'] = df['specter_input'].apply(self.generate_sparse_embedding)
        return df
