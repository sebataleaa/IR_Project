# dataset.py
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from csv_loader import CSVDataLoader
from inverted_index import InvertedIndex

class Datasets:
    def __init__(self):
        # Properly escape the special characters in strip_chars
        strip_chars = " .!?,@/\\#~:;'\""

        csv_loader1 = CSVDataLoader('lifestyle_dev/lifestyle_dev.csv', delimiter='\t',nrows=100)
        csv_loader2 = CSVDataLoader('wikIR1k/documents.csv', delimiter='\t',nrows=100)

        self.ID1 = InvertedIndex(
            csv_loader1,
            None,  # Doc2Vec model will be set later
            strip_chars,
            stopwords.words('english'),
            PorterStemmer(),
            WordNetLemmatizer()
        )

        self.ID2 = InvertedIndex(
            csv_loader2,
            None,
            strip_chars,
            stopwords.words('english'),
            PorterStemmer(),
            WordNetLemmatizer()
        )

    def query(self, dataset, query):
        if dataset == 'lifestyle_dev':
            return self.ID1.lookup(query)
        elif dataset == 'wikIR1k':
            return self.ID2.lookup(query)
        else:
            return None
