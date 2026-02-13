
import math
import re
import torch
import pandas as pd
from typing import List, Dict, Union, Tuple
from rapidfuzz import fuzz, utils
from sentence_transformers import CrossEncoder, SentenceTransformer, util

class EntityResolver:
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the heavy models into memory.
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Engine on {self.device.upper()}...")

        # 1. Load Bi-Encoder (Context/Retrieval)
        print("   - Loading Bi-Encoder (BAAI/bge-m3)...")
        self.bi_encoder = SentenceTransformer('BAAI/bge-m3', device=self.device)

        # 2. Load Cross-Encoder (Precision/Reranking)
        print("   - Loading Cross-Encoder (Unicamp mMarco)...")
        self.cross_encoder = CrossEncoder('unicamp-dl/mMiniLM-L6-v2-mmarco-v2', device=self.device)
        
        print("✅ System Ready.")

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Canonicalization layer: Removes legal suffixes and generic terms 
        to help the models focus on the core entity name.
        """
        if not isinstance(text, str): return ""
        
        text = text.upper().strip()
        
        # Remove Common Noise (Adjust based on your dataset)
        removals = [
            r"\bLTDA\b", r"\bS\.A\.\b", r"\bSA\b", r"\bME\b", r"\bEPP\b", 
            r"\bEIRELI\b", r"\-", r"\.", r"\bPOSTO\b", r"\bCOMERCIO\b"
        ]
        
        for pattern in removals:
            text = re.sub(pattern, "", text)
            
        return " ".join(text.split())

    def _calculate_sigmoid(self, logit: float) -> float:
        """Converts raw model logits to 0-100 probability."""
        return (1 / (1 + math.exp(-logit))) * 100

    def resolve_single(self, query: str, target: str) -> Dict:
        """
        Runs the full hybrid logic for a single pair.
        """
        # 0. Preprocessing
        clean_q = self.normalize_text(query)
        clean_t = self.normalize_text(target)

        # 1. Cross-Encoder Check (The Judge)
        cross_logits = self.cross_encoder.predict([(query, target)])
        cross_score = self._calculate_sigmoid(cross_logits[0])

        # 2. Bi-Encoder Check (The Lawyer)
        # Note: In production, you would pre-calculate target embeddings. 
        # Here we do real-time for demonstration.
        emb_q = self.bi_encoder.encode(query, convert_to_tensor=True)
        emb_t = self.bi_encoder.encode(target, convert_to_tensor=True)
        vector_score = util.cos_sim(emb_q, emb_t).item() * 100

        # 3. Fuzzy Check (The Scribe)
        fuzzy_score = fuzz.token_sort_ratio(clean_q, clean_t)

        # --- DECISION LOGIC (The "Brain") ---
        status = "❌ REJEITADO"
        method = "None"
        confidence = "Low"

        # Tier 1: Absolute Certainty (Formal/B2B)
        if cross_score > 90.0:
            status = "✅ APROVADO (AUTO)"
            method = "Cross-Encoder (Formal)"
            confidence = "High"
        
        # Tier 2: Strong Context (Nickname/Semantics) - Adjusted for Zé/José
        elif vector_score > 70.0:
            status = "⚠️ APROVAÇÃO CONDICIONAL"
            method = "Vector (Contexto)"
            confidence = "Medium"
            
        # Tier 3: Syntax Rescue (Typos)
        elif fuzzy_score > 85.0:
            status = "⚠️ APROVAÇÃO CONDICIONAL"
            method = "Fuzzy (Sintático)"
            confidence = "Medium"
            
        # Tier 4: Suspicious (Requires Human Eyes)
        elif vector_score > 60.0:
            status = "👀 REVISÃO HUMANA"
            method = "Possible Match"
            confidence = "Low"

        return {
            "Query": query,
            "Target": target,
            "Cross_Score": f"{cross_score:.1f}%",
            "Vector_Score": f"{vector_score:.1f}%",
            "Fuzzy_Score": f"{fuzzy_score:.1f}%",
            "Decision": status,
            "Primary_Method": method
        }

    def batch_resolve(self, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Processes a list of (query, target) tuples and returns a DataFrame.
        """
        results = [self.resolve_single(q, t) for q, t in pairs]
        return pd.DataFrame(results)

# --- EXECUTION BLOCK (DEMO) ---
if __name__ == "__main__":
    # Simulate a run
    engine = EntityResolver()

    test_cases = [
        ("APPLE COMPUTER BRA", "APPLE COMPUTER BRASIL LTDA"),
        ("POSTO SHELL AV BRASIL", "SHELL BRASIL LTDA"),
        ("RESTAURANTE DO ZE", "BAR E RESTAURANTE DO JOSE"), # The tricky one
        ("PADARIA DO JOAO", "PANIFICADORA SAO JOAO LTDA"),  # The semantic gap
        ("PAGAMENTO NF 12345", "AMBEV S.A.")                # The trash
    ]

    print("\n📊 Running Hybrid Matching Logic:\n")
    df = engine.batch_resolve(test_cases)
    
    # Clean output for terminal
    print(df.to_markdown(index=False))
    
    print("\nDone.")
