
# IR_Project 

## Intro

It's a basic ir system which contain multiple `documents` and an `api` to `query` these `documents`, So that it returns in the `response` the most `relevant documents` to the `query` according to many metrics.

## Features

- Data Processing: Cleans and tokenizes text data for better analysis
- Document Representation: Uses Doc2Vec to represent documents as dense vectors.
- Inverted Index: Efficient indexing of terms for fast query retrieval.
- Query Processing: Handles user queries and retrieves relevant documents.
- Ranking and Matching: Ranks documents based on similarity to the query.
- API Integration: Provides a FastAPI interface for easy access and querying. 
- Documents clustering.
- Use advanced word embedding models. 

## Project Structure (Classes)

### Text Tokenizer

It is a class that contain the full logic of converting a `document` to a list of needed `terms` only.

It takes in the constructor
(`strip_chars ,stop_words ,lemmatizer, stemmer`)

It is contain a functions that take a `text` as a param and returns a list of `terms`
These functions process text data by performing common preprocessing steps such as cleaning, stopword removal, normalizatin, POS tagging ..etc 

### Document

It is a class that contain the full needed information of a `document` like:

- document `id`
- original `text`
- list of document's `terms`
- `counter` for each `term` in this `document`
- unique set of `terms` 

we will explain it by using this variables :
   - `doc_id`: Stores the identifier of the document.
   - `text`: Stores the original text content of the document.
   - `tokens`: Stores the tokens extracted from the document.
   - `termCnt`: A `defaultdict` object that keeps track of the count of each term in the document. The count is initially set to 0 for all terms.
   - `uniqueTerms`: A set that stores the unique terms present in the document.
   - `lenTerms`: Stores the total number of tokens in the document.
   - `vector`: A NumPy array of size 20 (as an example), initialized with zeros. This vector is used to represent the document in a numerical form.

   ### Datasets

It is a class that contain the logic that deal with `ir_datasets` library, which is used to load the `datasets` and pass them with their information (eg.. `language`, `stemmer`, `lemmatizer`, `stopwords`) to the constructor of `InvertedIndex`.

Also it has a function that takes the `query` and the `dataset` and call the corresponding `dataset` with the `query` to get the relative documents from that `dataset`

- Note: The `api` call this function, And this function call another function in `InvertedIndex` class.

### InvertedIndex

this class represents an inverted index data structure and provides methods for building the index, querying documents, calculating cosine similarity, and evaluating query results, and considered the main class in the application which contains:

- Data structure that represent the `inverted index` of a collection of `documents` and their `terms`.
- The `TF` ,`IDF` ,`TF_IDF` , `queries` , ... etc information.
- It is has 4 functions:

  1- build():

  - It takes the `dataset` and iterate over `documents` to build and fill the data structure with `dataset` information.

  2- cosine_similarity():

  - It calculates the cosine similarity between two vectors by calculating their dot product and dividing it by the product of their norms. It provides a measure of similarity between the vectors.

  3- evaluate():

  - It calculates various evaluation metrics (precision, recall, average precision, mean average precision, and mean reciprocal rank) for a given query. It compares the retrieved documents with the relevant documents, updates the evaluation metrics accordingly, and returns the calculated metrics in a dictionary.

  4- lookup():

  - It takes an input query, tokenizes it, infers the query vector, calculates cosine similarity between the query vector and document vectors, retrieves relevant documents based on the similarity threshold, sorts the results by similarity, and returns the top-k retrieved documents along with evaluation metrics.




## Languages, Frameworks & Libraries

    Front-end Application:
    - Flutter

    Back-end Application:
    - Python
    - FastAPI
    - NLTK
    - IR_datasets
    - Contractions

---

## Development Team

    - Mohammed Hwaidi
    - Seba Taleaa
    - Lilian Kabool
    - Laila Oudah
    - Yumna Qassuma