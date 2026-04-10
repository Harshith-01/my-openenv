import re
from typing import List, Dict, Any

import numpy as np

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    total = len(tokens)
    if total == 0:
        return tf

    for word in tokens:
        tf[word] = tf.get(word, 0.0) + 1.0

    for word in tf:
        tf[word] = tf[word] / total

    return tf


class SemanticKnowledgeBase:
    def __init__(self):
        self.documents = [
            {
                "id": "doc_001",
                "title": "Refund Policy Overview",
                "chunk": "Standard refund policy dictates that for broken hardware components, we issue a replacement or refund within 30 days of purchase. The agent must verify the serial number.",
            },
            {
                "id": "doc_002",
                "title": "Hardware Troubleshooting - GPU",
                "chunk": "If a user reports screen flickering or boot failures related to the GPU, check order history. If the model is GTX-9990 and recently ordered, it is a known manufacturing defect. Issue a refund or replacement.",
            },
            {
                "id": "doc_003",
                "title": "API 500 Errors - Keigo and Escalation Protocol",
                "chunk": "For Japanese enterprise clients experiencing API 500 internal server errors, this is usually caused by backend bug ID #ERR-7782 (rate limit race condition). Agents must escalate immediately using a Kaizen-style report. The response should include formal Japanese Keigo, such as '\u7533\u3057\u8a33\u3054\u3056\u3044\u307e\u305b\u3093', and the company reference '\u5f0a\u793e'.",
            },
            {
                "id": "doc_004",
                "title": "Password Reset Procedures",
                "chunk": "For password reset requests, categorize the ticket as 'account' and apply the 'password_reset' tag. No extra operations are required.",
            },
        ]
        self._build_index()

    def _build_index(self) -> None:
        self.idf: Dict[str, float] = {}
        total_docs = len(self.documents)
        doc_tokens: List[List[str]] = []

        for doc in self.documents:
            tokens = tokenize(doc["title"] + " " + doc["chunk"])
            doc_tokens.append(tokens)
            for token in set(tokens):
                self.idf[token] = self.idf.get(token, 0.0) + 1.0

        for token in self.idf:
            self.idf[token] = np.log(total_docs / (self.idf[token] + 1.0))

        self.doc_vectors: List[Dict[str, float]] = []
        for tokens in doc_tokens:
            tf = compute_tf(tokens)
            vec = {token: tf[token] * self.idf.get(token, 0.0) for token in tf}
            self.doc_vectors.append(vec)

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[token] * vec2[token] for token in intersection)
        sum1 = sum(value ** 2 for value in vec1.values())
        sum2 = sum(value ** 2 for value in vec2.values())
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator / denominator)

    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        tf = compute_tf(tokens)
        query_vec = {token: tf[token] * self.idf.get(token, np.log(len(self.documents))) for token in tf}

        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((score, self.documents[i]))

        scores.sort(key=lambda item: item[0], reverse=True)
        results = [doc for score, doc in scores[:top_k] if score > 0.05]
        if not results:
            return [{"id": "none", "title": "No Results", "chunk": "No relevant documents found in knowledge base."}]
        return results


class MockDatabase:
    def __init__(self):
        self.users = {
            "u100": {"name": "Alice Johnson", "email": "alice@example.com"},
            "u101": {"name": "Kenji Sato", "email": "kenji.sato@enterprise.co.jp"},
            "u102": {"name": "Bob Smith", "email": "bob@example.com"},
        }
        self.orders = {
            "o992": {"user_id": "u100", "product": "GTX-9990 GPU", "date": "2023-10-01", "serial": "SN-GPU-8819"},
            "o993": {"user_id": "u102", "product": "Standard Keyboard", "date": "2023-01-15", "serial": "SN-KB-112"},
        }

    def query(self, query_str: str) -> str:
        query_str_lower = query_str.lower()
        if any(token in query_str_lower for token in ("u100", "alice")) and "order" not in query_str_lower:
            return str(self.users["u100"])
        if any(token in query_str_lower for token in ("u101", "kenji")) and "order" not in query_str_lower:
            return str(self.users["u101"])
        if any(token in query_str_lower for token in ("order", "gpu", "serial", "sn-gpu-8819", "o992")):
            return str(self.orders["o992"])
        return "No matching records found."


knowledge_base = SemanticKnowledgeBase()
customer_db = MockDatabase()
