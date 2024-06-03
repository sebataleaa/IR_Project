# # csv_loader.py
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from document import Document

# class CSVDataLoader:
#     def __init__(self, doc_file, delimiter='\t',nrows=100):
#         # Read CSV files with better error handling using on_bad_lines
#         self.documents = pd.read_csv(doc_file, delimiter=delimiter, header=None, quoting=3, engine='python')
    
#     def docs_iter(self):
#         # Adjust columns to match the actual column names in your CSV
#         for index, row in self.documents.iterrows():
#             yield Document(doc_id=row[0], text=row[1], terms=word_tokenize(row[1].lower()))
# csv_loader.py
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from document import Document
import pandas as pd
from nltk.tokenize import word_tokenize
from document import Document

class CSVDataLoader:
    def __init__(self, doc_file, dataset_type='wikIR1k', delimiter=',', nrows=10000):
        self.dataset_type = dataset_type
        
        if self.dataset_type == 'wikIR1k':
            self.documents = pd.read_csv(doc_file, delimiter=delimiter, nrows=nrows)
            self.queries = pd.read_csv('wikIR1k/test/queries.csv', delimiter=delimiter)
            self.qrels = pd.read_csv('wikIR1k/test/BM25.qrels.csv', delimiter=delimiter)
        elif self.dataset_type == 'lifestyle_dev':
            self.documents = pd.read_csv(doc_file, delimiter=delimiter, header=None, quoting=3, engine='python', nrows=nrows)
            self.queries = None
            self.qrels = None
        
        print("Documents DataFrame head:")
        print(self.documents.head())  # Print the first few rows to inspect the structure
        if self.queries is not None:
            print("Queries DataFrame head:")
            print(self.queries.head())
        if self.qrels is not None:
            print("Qrels DataFrame head:")
            print(self.qrels.head())

    def docs_iter(self):
        for index, row in self.documents.iterrows():
            if self.dataset_type == 'wikIR1k':
                yield Document(doc_id=str(row['id_right']), text=row['text_right'].strip(), tokens=word_tokenize(row['text_right'].lower()))
            elif self.dataset_type == 'lifestyle_dev':
                yield Document(doc_id=row[0], text=row[1], tokens=word_tokenize(row[1].lower()))

    def qrels_iter(self):
        if self.dataset_type == 'wikIR1k' and self.qrels is not None:
            for index, row in self.qrels.iterrows():
                yield (row['id_left'], row['id_right'], row['label'])
        else:
            # Placeholder if qrels data is not available or for other dataset types
            return iter([])

    def queries_iter(self):
        if self.dataset_type == 'wikIR1k' and self.queries is not None:
            for index, row in self.queries.iterrows():
                yield (row['id_left'], row['text_left'])
        else:
            # Placeholder if queries data is not available or for other dataset types
            return iter([])

