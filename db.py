import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# We provide a mock/simple TF-IDF or embedding vectorizer to represent semantic search without
# relying on downloading huge models on every HF space reload.
# In a real production system, this could be replaced by `sentence-transformers`.
# For the hackathon & resume, we'll implement a clean TF-IDF based semantic similarity to keep
# the docker image light and extremely fast, while demonstrating the "chunking" and "semantic search" architecture.

def tokenize(text: str) -> List[str]:
    # Simple semantic tokenization
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def compute_tf(tokens: List[str]) -> Dict[str, float]:
    tf = {}
    total = len(tokens)
    for word in tokens:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] = tf[word] / total
    return tf

class SemanticKnowledgeBase:
    def __init__(self):
        # We perform semantic chunking. Rather than random splits, we split by logical document sections.
        self.documents = [
            {
                "id": "doc_001",
                "title": "Refund Policy Overview",
                "chunk": "Standard refund policy strictly dictates that for broken hardware components, we issue a replacement or refund only within 30 days of purchase. The agent must verify the Serial Number."
            },
            {
                "id": "doc_002",
                "title": "Hardware Troubleshooting - GPU",
                "chunk": "If a user reports screen flickering or boot failures related to the GPU, check the order history. If the GPU model is 'GTX-9990' and it was ordered recently, it is a known manufacturing defect. Issue a refund."
            },
            {
                "id": "doc_003",
                "title": "API 500 Errors - Keigo & Escalation Protocol",
                "chunk": "For Japanese Enterprise Clients experiencing API 500 internal server errors, this is usually caused by Backend Bug ID #ERR-7782 (Rate limit race condition). Agents must escalate this issue immediately using the Kaizen Report template. The agent must politely apologize to the user using formal Japanese Keigo (e.g., '申し訳ございません' - we deeply apologize, and '弊社' - our company)."
            },
            {
                "id": "doc_004",
                "title": "Password Reset Procedures",
                "chunk": "For password resets, the ticket should be categorized strictly as 'account' and given the tag 'password_reset'. No further action is required from the agent as the system handles the email automatically."
            }
        ]
        self._build_index()

    def _build_index(self):
        # Build a simple inverted index & IDF for our mock semantic search
        self.idf = {}
        total_docs = len(self.documents)
        doc_tokens = []
        for doc in self.documents:
            tokens = tokenize(doc["title"] + " " + doc["chunk"])
            doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.idf[t] = self.idf.get(t, 0) + 1
        
        for t in self.idf:
            self.idf[t] = np.log(total_docs / (self.idf[t] + 1))
            
        self.doc_vectors = []
        for tokens in doc_tokens:
            tf = compute_tf(tokens)
            vec = {t: tf[t] * self.idf.get(t, 0) for t in tf}
            self.doc_vectors.append(vec)

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator / denominator)

    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        tf = compute_tf(tokens)
        query_vec = {t: tf[t] * self.idf.get(t, np.log(len(self.documents))) for t in tf}
        
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((score, self.documents[i]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        # Only return if there is some relevance
        results = [doc for score, doc in scores[:top_k] if score > 0.05]
        if not results:
            return [{"id": "none", "title": "No Results", "chunk": "No relevant documents found in knowledge base."}]
        return results

class MockDatabase:
    def __init__(self):
        self.users = {
            "u100": {"name": "Alice Johnson", "email": "alice@example.com"},
            "u101": {"name": "Kenji Sato", "email": "kenji.sato@enterprise.co.jp"},
            "u102": {"name": "Bob Smith", "email": "bob@example.com"}
        }
        self.orders = {
            "o992": {"user_id": "u100", "product": "GTX-9990 GPU", "date": "2023-10-01", "serial": "SN-GPU-8819"},
            "o993": {"user_id": "u102", "product": "Standard Keyboard", "date": "2023-01-15", "serial": "SN-KB-112"}
        }
    
    def query(self, query_str: str) -> str:
        # A simple simulated SQL/Text-based query
        query_str_lower = query_str.lower()
        if "user" in query_str_lower and "alice" in query_str_lower:
            return str(self.users["u100"])
        elif "user" in query_str_lower and "kenji" in query_str_lower:
            return str(self.users["u101"])
        elif "order" in query_str_lower or "gpu" in query_str_lower:
            return str(self.orders["o992"])
        else:
            return "No matching records found."

# Singletons for the environment
knowledge_base = SemanticKnowledgeBase()
customer_db = MockDatabase()
