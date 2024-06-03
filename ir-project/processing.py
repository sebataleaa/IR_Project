import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV

class TextTokenizer:
    def __init__(self, strip_chars=None, stop_words=None, lemmatizer=None, stemmer=None):
        self.strip_chars = strip_chars if strip_chars else ""
        self.stop_words = set(stop_words) if stop_words else set(stopwords.words('english'))
        self.lemmatizer = lemmatizer if lemmatizer else WordNetLemmatizer()
        # self.stemmer = stemmer if stemmer else PorterStemmer()  # Optional: Uncomment for stemming

    def clean_text(self, text):
        # Remove digits and punctuation, convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        return text

    def remove_stopwords(self, text):
        return ' '.join(word for word in text.split() if word not in self.stop_words)

    def normalize_text(self, text):
        # Add any specific normalization steps here if needed
        # Example: text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return ADJ
        elif treebank_tag.startswith('V'):
            return VERB
        elif treebank_tag.startswith('N'):
            return NOUN
        elif treebank_tag.startswith('R'):
            return ADV
        else:
            return NOUN

    def tokenize_text(self, text):
        # Normalize the text
        normalized_text = self.normalize_text(text)
        # Clean the text
        cleaned_text = self.clean_text(normalized_text)
        # Remove stopwords
        no_stopwords_text = self.remove_stopwords(cleaned_text)
        # Tokenize
        word_tokens = word_tokenize(no_stopwords_text)
        # Apply POS tagging
        pos_tagged_tokens = pos_tag(word_tokens)
        # Lemmatize tokens with POS tagging
        lemmatized_tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in pos_tagged_tokens if word not in self.stop_words]
        # Apply stemming (optional)
        # stemmed_tokens = [self.stemmer.stem(word) for word in lemmatized_tokens]
        return lemmatized_tokens

    def process_query(self, query):
        # Normalize, clean, tokenize, and lemmatize/stem the query
        query_tokens = self.tokenize_text(query)
        return query_tokens
