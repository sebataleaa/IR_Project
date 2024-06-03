import numpy as np
from collections import defaultdict

class Document:
    def __init__(self, doc_id, text, tokens):
        self.doc_id = doc_id
        self.text = text
        self.tokens = tokens
        self.termCnt = defaultdict(int)
        self.uniqueTerms = set(tokens)
        self.lenTerms = len(tokens)
        self.vector = np.zeros(20)  # Example vector size, adjust as needed

        for term in tokens:
            self.termCnt[term] += 1

    def insertToVec(self, tfidf_value, term):
        self.vector += tfidf_value  # Simple summation, customize as needed

    def set_vector(self, vector):
        self.vector = vector
