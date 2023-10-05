from nltk.stem import PorterStemmer
import nltk
from sklearn.metrics.pairwise import linear_kernel
import string
from nltk.corpus import stopwords
import math
from collections import Counter , defaultdict
from tqdm.notebook import tqdm
from typing import List
import numpy as np

#nltk.download('stopwords')
#stemmer = PorterStemmer()

# Tokenize, stem a document
def tokenize(text, stemmer):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stemmed = " ".join([stemmer.stem(word.lower()) for word in tokens if word not in stopwords.words('english')])
    return stemmed.split()

# compute IDF, storing idf values in a dictionary
def idf_values(vocabulary, documents):
    idf = {}
    num_documents = len(documents)
    for i, term in enumerate(vocabulary):
        idf[term] = math.log(num_documents/sum(term in document for document in documents), math.e)
    return idf

# Function to generate the vector for a document (with normalisation)
def vectorize(document, vocabulary, idf):
    vector = [0]*len(vocabulary)
    counts = Counter(document)
    max_count = counts.most_common(1)[0][1]
    for i,term in enumerate(vocabulary):
        vector[i] = idf[term] * counts[term]/max_count
    return vector

# Define a function to compute the document frequencies for a subset of the vocabulary
def compute_doc_freqs(vocab_subset, documents, id):
    doc_freqs = defaultdict(int)
    for word in tqdm(vocab_subset, desc=f"Process {id}"):
        for doc in documents:
            if word in doc:
                doc_freqs[word] += 1
    return doc_freqs

def count_terms(document: List[str]):
    """
        Takes a document and returns:
            - terms: dict(term: count)
            - max_count: int
        
        
        terms is a dictionary that maps each term to its count in the document.
        max_count is the count of the most frequent term in the document.
    """
    terms = Counter(document)
    max_count = terms.most_common(1)[0][1]
    return terms, max_count

def compute_query_vector(query,vocabulary_dict,idfs):
    query_vec = np.zeros((len(vocabulary_dict)))
    counts, max_count = count_terms(query)
    for term, count in counts.items():
        if term in vocabulary_dict:
            term_id = vocabulary_dict[term]
            query_vec[term_id] = count/max_count * idfs[term]
    return query_vec