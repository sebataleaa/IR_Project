from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    dataset_name: str

class DocumentResponse(BaseModel):
    document_id: int
    similarity: float
    text: str

class QueryResponse(BaseModel):
    data: List[DocumentResponse]

class FastAPIApp:
    def __init__(self, datasets):
        self.datasets = datasets
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/search", response_model=QueryResponse)
        def search(query_request: QueryRequest):
            query = query_request.query
            top_k = query_request.top_k
            dataset_name = query_request.dataset_name
            
            inverted_index = self.datasets.get(dataset_name)
            if not inverted_index:
                return QueryResponse(data=[])
            
            results = inverted_index.lookup(query, top_k=top_k)
            response = []
            for result in results['irsResult']:
                doc_id = result['doc_id']
                similarity = result['doc_similarity']
                doc_text = inverted_index.df[inverted_index.df['id'] == doc_id]['text'].values[0]
                response.append(DocumentResponse(document_id=doc_id, similarity=similarity, text=doc_text))
            return QueryResponse(data=response)

    def run(self, host="127.0.0.1", port=8001):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)