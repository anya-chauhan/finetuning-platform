# app/services/data_service.py
import torch
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from app.core.config import settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DataService:
    """Service for managing PINNACLE data and embeddings"""
    
    def __init__(self):
        self.pinnacle_embeddings_dict: Optional[Dict] = None
        self.protein_ids: List[str] = []
        self.context_ids: List[int] = []
        self.context_names_map: Dict[int, str] = {}
        self.celltype_protein_dict: Dict[str, List[str]] = {}
        self.num_proteins: int = 0
        self.num_contexts: int = 0
        self.embedding_dim: int = 0
        
    async def initialize(self):
        """Initialize PINNACLE data"""
        await self.load_pinnacle_data()
        
    async def load_pinnacle_data(self):
        """Load PINNACLE data following their format"""
        try:
            # Load embeddings
            embed_data = torch.load(settings.EMBEDDINGS_PATH, map_location='cpu')
            
            # Get context IDs
            self.context_ids = sorted(list(embed_data.keys()))
            print(f"Found {len(self.context_ids)} contexts")
            
            # Get embedding dimension
            self.embedding_dim = embed_data[self.context_ids[0]].shape[1]
            
            # Load and parse labels
            with open(settings.LABELS_PATH, "r") as f:
                labels_dict = f.read()
            labels_dict = labels_dict.replace("\'", "\"")
            labels_dict = json.loads(labels_dict)
            
            # Extract CCI cell types
            celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
            celltype_names = [ct.split("CCI_")[1] for ct in celltypes]
            celltype_dict = {name: i for i, name in enumerate(celltype_names)}

            # Create context ID to name mapping
            self.context_names_map = {i: name for name, i in celltype_dict.items()}

            # Verify all context IDs have names
            for cid in self.context_ids:
                if cid not in self.context_names_map:
                    print(f"Warning: Context ID {cid} has no matching cell type name")
                    self.context_names_map[cid] = f"unknown_context_{cid}"

            # Extract proteins
            protein_names = []
            protein_celltypes = []
            for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
                if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"):
                    continue
                protein_names.append(p)
                protein_celltypes.append(c)
            
            print(f"Found {len(protein_names)} protein cell type associations")
            print(f"Found {len(celltypes)} CCI cell types")
            
            # Build celltype_protein_dict
            proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
            self.celltype_protein_dict = proteins.pivot_table(
                values="target", 
                index="cell type", 
                aggfunc={"target": list}
            ).to_dict()["target"]

            # Debug info
            print("\nAnalyzing context-protein relationships:")
            for i in range(min(3, len(self.context_ids))):
                context_id = self.context_ids[i]
                context_name = self.context_names_map.get(context_id, f"unknown_{context_id}")
                embeddings_shape = embed_data[context_id].shape
                
                print(f"\nContext {context_id} ('{context_name}'):")
                print(f"  Embeddings shape: {embeddings_shape}")
                
                if context_name in self.celltype_protein_dict:
                    proteins_for_context = self.celltype_protein_dict[context_name]
                    print(f"  Proteins in celltype_protein_dict: {len(proteins_for_context)}")
                    print(f"  First 5 proteins: {proteins_for_context[:5]}")
            
            # Store unique proteins
            self.protein_ids = list(set(protein_names))
            
            # Store embeddings
            self.pinnacle_embeddings_dict = embed_data
            
            # Set counts
            self.num_proteins = len(self.protein_ids)
            self.num_contexts = len(self.context_ids)
            
            print(f"✅ Data loaded: {self.num_proteins} proteins, {self.num_contexts} contexts, {self.embedding_dim}D embeddings")
            
        except Exception as e:
            print(f"❌ Error loading PINNACLE data: {e}")
            raise

    def get_context_names(self) -> List[str]:
        """Get list of context names in order"""
        return [self.context_names_map.get(cid, f"unknown_{cid}") for cid in self.context_ids]
    
    def get_proteins(self, search: Optional[str] = None, limit: int = 100) -> Dict:
        """Get available proteins with optional search"""
        if search:
            filtered = [p for p in self.protein_ids if search.lower() in p.lower()]
            return {
                "proteins": filtered[:limit],
                "total_matches": len(filtered),
                "total_proteins": len(self.protein_ids)
            }
        else:
            return {
                "proteins": self.protein_ids[:limit],
                "showing": min(limit, len(self.protein_ids)),
                "total_proteins": len(self.protein_ids)
            }
    
    def parse_context(self, context: str) -> int:
        """Parse context string to context ID"""
        try:
            context_id = int(context)
            if context_id not in self.pinnacle_embeddings_dict:
                raise ValueError(f"Context ID {context_id} not found")
            return context_id
        except ValueError:
            # Try to find by name
            for cid, cname in self.context_names_map.items():
                if cname == context:
                    return cid
            raise ValueError(f"Context '{context}' not found")
    
    def get_embeddings_for_proteins(self, protein_ids: List[str], context_id: int) -> Tuple[List[torch.Tensor], List[int]]:
        """Get embeddings for specific proteins in a context"""
        context_name = self.context_names_map.get(context_id)
        if context_name not in self.celltype_protein_dict:
            raise ValueError(f"No protein list found for context {context_name}")
        
        context_proteins = self.celltype_protein_dict[context_name]
        context_embeddings = self.pinnacle_embeddings_dict[context_id]
        
        embeddings_list = []
        valid_indices = []
        
        for i, protein_id in enumerate(protein_ids):
            try:
                protein_idx = context_proteins.index(protein_id)
                embedding = context_embeddings[protein_idx]
                embeddings_list.append(embedding)
                valid_indices.append(i)
            except ValueError:
                print(f"Warning: Protein '{protein_id}' not found in context {context_name}")
                continue
        
        return embeddings_list, valid_indices
    
    def find_similar_proteins(self, protein_id: str, context_id: int, top_k: int = 10) -> List[Dict]:
        """Find most similar proteins in a context"""
        context_name = self.context_names_map[context_id]
        context_proteins = self.celltype_protein_dict.get(context_name, [])
        context_embeddings = self.pinnacle_embeddings_dict[context_id].numpy()
        
        if protein_id not in context_proteins:
            raise ValueError(f"Protein {protein_id} not found in context {context_name}")
        
        query_idx = context_proteins.index(protein_id)
        query_embedding = context_embeddings[query_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, context_embeddings)[0]
        
        # Get top K similar proteins (excluding query)
        top_indices = np.argsort(similarities)[::-1][:top_k+1]
        
        results = []
        for idx in top_indices:
            if idx != query_idx:
                results.append({
                    "protein_id": context_proteins[idx],
                    "similarity": float(similarities[idx])
                })
        
        return results[:top_k]
    
    def get_protein_contexts(self, protein_id: str) -> List[Dict]:
        """Get all contexts where a protein appears"""
        contexts_with_protein = []
        
        for context_id in self.context_ids:
            context_name = self.context_names_map.get(context_id)
            context_proteins = self.celltype_protein_dict.get(context_name, [])
            
            if protein_id in context_proteins:
                contexts_with_protein.append({
                    "context_id": context_id,
                    "context_name": context_name,
                    "protein_count": len(context_proteins)
                })
        
        return contexts_with_protein