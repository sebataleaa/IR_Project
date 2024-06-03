from collections import defaultdict
import math
import numpy as np
from processing import TextTokenizer
from gensim.models.doc2vec import Doc2Vec
from document import Document

class InvertedIndex:
    def __init__(self, dataset, model, strip_chars, stop_words=None, stemmer=None, lemmatizer=None):
        self.documents = {}
        self.dataset = dataset
        self.model = model

        self.idf = defaultdict(float)
        self.query_docs = defaultdict(set)
        self.tf = defaultdict(lambda: defaultdict(float))
        self.tf_idf = defaultdict(lambda: defaultdict(float))
        self.invertedIndex = defaultdict(lambda: defaultdict(bool))
        self.textTokenizer = TextTokenizer(strip_chars, stop_words, lemmatizer, stemmer)

        self.sumRR = 0.0
        self.sumAvp = 0.0
        self.queryCnt = 0
        self.threshold = 0.3

        if self.model is not None:
            self.build()

    def build(self):
        print("Start Building Inverted Index:")

        n_docs = 0
        for doc in self.dataset.docs_iter():
            n_docs += 1
            tokens = self.textTokenizer.tokenize_text(doc.text)
            document = self.documents[doc.doc_id] = Document(doc.doc_id, doc.text, tokens)
            doc_vector = self.model.infer_vector(tokens)
            document.set_vector(doc_vector)
            
            for term in document.uniqueTerms:
                self.invertedIndex[term][document.doc_id] = True
                self.tf[term][document.doc_id] = math.log(1.0 + document.termCnt[term] / document.lenTerms)

        for term, doc_ids in self.invertedIndex.items():
            self.idf[term] = math.log(n_docs / len(doc_ids))

        for term, docFrequency in self.tf.items():
            for doc_id, tf in docFrequency.items():
                self.tf_idf[term][doc_id] = tf * self.idf[term]
                self.documents[doc_id].insertToVec(self.tf_idf[term][doc_id], term)

        print("Inverted Index Built Successfully.")

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    def evaluateQuery(self, irsResult, query, queryId=None):
        if queryId is None:
            queryId = 0
            for query_id, text in self.dataset.queries_iter():
                if query == text:
                    queryId = query_id
                    break

      
        precisionSum = 0.0
        numRelevantRetrieved = 0
        numRetrieved = len(irsResult)
        numRelevant = len(self.query_docs[queryId])

      

        firstRelevantIsFound = False
        for i, res in enumerate(irsResult):
            if res['doc_id'] in self.query_docs[queryId]:
                if not firstRelevantIsFound:
                    self.sumRR += 1.0 / (i + 1.0)
                    firstRelevantIsFound = True
                numRelevantRetrieved += 1
                precisionSum += numRelevantRetrieved / (i + 1)

        precision = numRelevantRetrieved / numRetrieved if numRetrieved != 0 else 0
        recall = numRelevantRetrieved / numRelevant if numRelevant != 0 else 0
        avp = precisionSum / numRetrieved if numRetrieved != 0 else 0

        self.queryCnt += 1
        self.sumAvp += avp
        mapVal = self.sumAvp / self.queryCnt
        mrr = self.sumRR / self.queryCnt


        return {
            "precision": precision,
            "recall": recall,
            "avp": avp,
            "map": mapVal,
            "mrr": mrr
        }

    def lookup(self, inputQuery, queryId=None, top_k=5):
        query_tokens = self.textTokenizer.tokenize_text(inputQuery)
        query_vector = self.model.infer_vector(query_tokens)
        irsResult = []
        sim = defaultdict(float)

        for document in self.documents.values():
            doc_similarity = self.cosine_similarity(query_vector, document.vector)
            if doc_similarity > self.threshold:
                irsResult.append({
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "doc_similarity": doc_similarity
                })

        irsResult.sort(key=lambda x: x['doc_similarity'], reverse=True)

        if len(irsResult) == 0:
            # Fallback to similar documents if no results found
            fallback_docs = sorted(self.documents.values(), key=lambda doc: self.cosine_similarity(query_vector, doc.vector), reverse=True)[:top_k]
            irsResult = [{
                "doc_id": doc.doc_id,
                "text": doc.text,
                "doc_similarity": self.cosine_similarity(query_vector, doc.vector)
            } for doc in fallback_docs]

        return {
            "irsResult": irsResult[:top_k],
            "evaluation": self.evaluateQuery(irsResult, inputQuery, queryId)
        }