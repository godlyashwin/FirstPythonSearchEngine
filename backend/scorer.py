# GOAL:
# To come back with a set of documents, 
# Sorted their relevance to the user’s query
# Aim to have the most relevant results on top of the page

# To get the documents from a web crawler
from .dynamicUrl import *
from .dynamicUrl import progress, getProgress
import threading
import time
import logging
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access environment variables
API_KEY = os.getenv("API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
# To quantify how a particular document is relevant to a given query
# Count the number of overlapping terms
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

# To convert all the vocabulary within the documents into vectors
# Then map those vectors to the frequencies of specific terms
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

# Ignore warnings
import warnings

# Global progress variable
_progress_lock = threading.Lock()

#for i, paragraph in enumerate(docs, 1):
#        print(f"Paragraph {i}: {paragraph}")
if __name__ == "__main__":
    query = input("Enter your query: ")
    numResults = int(input("How many sources do you want to see?: "))

# Main search function
def search(query, numResults):
    global progress
    docs = []
    sources = []
    
    # Reset progress
    with _progress_lock:
        progress = {"progress": 0, "status": "Processing", "results": None}
    print("Progress: " + progress["status"])
    
    # Start the main function in a thread with error handling
    class SearchThread(threading.Thread):
        def __init__(self, target, args):
            super().__init__(target=target, args=args)
            self.exception = None
        
        def run(self):
            try:
                super().run()
            except Exception as e:
                self.exception = e
    try:
        # Validate environment variables
        if not API_KEY or not SEARCH_ENGINE_ID:
            raise ValueError("API_KEY or SEARCH_ENGINE_ID not set in .env file")
        findContent = SearchThread(target=search_dynamic_content, args=(query, docs, sources, API_KEY, SEARCH_ENGINE_ID))
        findContent.start()
        
        # Repeatedly check progress until the main function ends or times out
        progress_updates = []
        previousProgress = None
        start_time = time.time()
        timeout = 30  # Timeout after 30 seconds
        
        while findContent.is_alive():       
            progress = getProgress()
            if previousProgress is None or previousProgress["progress"] != progress["progress"]:
                previousProgress = progress
                start_time = time.time() # Restarting timer
                print(f"Progress: {progress['progress']}% - {progress['status']}")
                progress_updates.append(progress["progress"])
            
            time.sleep(0.5)
        
        # Ensure the main thread is complete
        findContent.join()
        
        # Check for exceptions
        if findContent.exception:
            print(f"search_dynamic_content failed with: {findContent.exception}")
            raise findContent.exception
        
        print("findContent finished!")
    except Exception as e:
        logging.error(f"Error in start_search: {e}")
        progress["results"] = ["Error processing search."]
    if docs:
        # TOKENIZATION: 
        # Splits text into normalized tokens (lowercase, remove punctuation, filtering stop words)
        REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
        TOKENIZER = TreebankWordTokenizer()

        # STEMMING: 
        # Stripping words of their plural forms, formatting, etc. to their "stem"
        # Makes it easy to compare words with those in the documents

        STEMMER = PorterStemmer()

        # Turns the user query into a list of terms
        def tokenize_and_stem(s):
            return [STEMMER.stem(t) for t 
                    in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

        # Consider texts not as lists of terms
        # But rather as numerical vectors in one whole vector space

        # Gather the vocabulary across the entire collection of documents
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
        vectorizer.fit(docs)
        progress = {"progress": 92, "status": "Tokenizing and stemming the query", "results": None}
        print(f"Progress: {progress}%")  # Optional: Display progress
        # Every term now is mapped to a index in a vector

        # Transform texts into vectors
        query_vector = np.asarray(vectorizer.transform([query]).todense())

        # Now, the score can be considered a similarity of these two representations
        # Vectors with similar directions will represent terms considered similar
        doc_vectors = np.asarray(vectorizer.transform(docs).todense())

        # Calculate cosine distance between the query vector and all vectors for all documents
        similarity = cosine_similarity(np.asarray(query_vector), np.asarray(doc_vectors))
        progress = {"progress": 95, "status": "Comparing documents to the query", "results": None}
        print(f"Progress: {progress}%")  # Optional: Display progress
        # Rank documents by their scores
        ranks = (-similarity).argsort(axis=None)
        progress = {"progress": 96, "status": "Ranking documents", "results": None}
        print(f"Progress: {progress}%")  # Optional: Display progress

        # A dictionary showing feedback on which results turned out relevant and which not
        # 'Querys': (Document Index, Feedback Value)
        # Feedback Value: 1. if relevant, 0. if not

        feedback = {
                'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)],
                'about page': [(0, 1.)]
        }

        # If this was given: 'who makes chatbots' and 'about page'
        # It would give the document about the chatbot
        # But what about the 'about page'?

        # Improve search performance with feedback using machine learning
        # Done by optimising feature weights from the score formula

        # Given a particular query, look if this query was asked before

        # Define feedback features: positive and negative
        # Positive means # of times a document was relevant for a query
        # Negative means # of times a document was irrelevant for a query

        # If a query wasn't asked before, find the nearest neighbor query
        # Then find feedback of that query
        # Weigh the two feedback features by the similarity

        # If a query which is very similar to the given one is found
        # Feedback features impact the overall score a lot
        # Otherwise the features are close to zero

        # Score each document with a proportion of postive feedback they received

        # For a given query, find a nearest neighbor that does have feedback
        progress = {"progress": 97, "status": "Attaining and using feedback", "results": None}
        print(f"Progress: {progress}%")  # Optional: Display progress
        feedback_queries = list(feedback.keys())

        similarity = cosine_similarity(vectorizer.transform([query]), 
                                    vectorizer.transform(feedback_queries))
        progress = {"progress": 97, "status": "Attaining and using feedback", "results": None}

        max_idx = np.argmax(similarity)

        # Look at all documents that received positive feedback
        pos_feedback_doc_idx = [idx for idx, feedback_value 
                                in feedback[feedback_queries[max_idx]] 
                                if feedback_value == 1.]


        counts = Counter(pos_feedback_doc_idx)

        # Find the proportion of positive feedback for each documents
        pos_feedback_proportions = {
                doc_idx: count / sum(counts.values()) for doc_idx, count in counts.items()
        }

        # Scale with a similarity between the original query and nearest neighbor query
        # Output a feature vector of it
        nn_similarity = np.max(similarity)
        pos_feedback_feature = [nn_similarity * pos_feedback_proportions.get(idx, 0.) 
                                for idx, _ in enumerate(docs)]

        # Summarized Code

        class Scorer:
            """Scores documents for a search query based on tf-idf similarity and relevance feedback"""
            def __init__(self, docs):
                """Initialize a scorer with a collection of documents, fit a vectorizer and list feature functions"""
                self.docs = docs
                # Suppress TfidfVectorizer stop_words warning
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
                self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
                self.doc_tfidf = self.vectorizer.fit_transform(docs)
                
                self.features = [
                    self._feature_tfidf,
                    self._feature_positive_feedback,
                ]
                self.feature_weights = [
                    1.,
                    2.,
                ]
                
                self.feedback = {}
                
            def score(self, query):
                """Generic scoring function: output a numpy array of scores aligned with document list"""
                feature_vectors = [feature(query) for feature in self.features]
                feature_vectors_weighted = [feature * weight for feature, weight in zip(feature_vectors, self.feature_weights)]
                return np.sum(feature_vectors_weighted, axis=0)
            
            def learn_feedback(self, feedback_dict):
                """Learn feedback in the form of `query` -> (doc index, feedback value)"""
                self.feedback = feedback_dict
                
            def _feature_tfidf(self, query):
                """TF-IDF feature: cosine similarities between TF-IDF vectors of documents and query"""
                query_vector = np.asarray(self.vectorizer.transform([query]).todense())
                similarity = cosine_similarity(query_vector, np.asarray(self.doc_tfidf.todense()))
                return similarity.ravel()
            
            def _feature_positive_feedback(self, query):
                """Positive feedback feature: assign positive values based on feedback"""
                if not self.feedback:
                    return np.zeros(len(self.docs))
                
                feedback_queries = list(self.feedback.keys())
                query_vector = np.asarray(self.vectorizer.transform([query]).todense())
                feedback_vectors = np.asarray(self.vectorizer.transform(feedback_queries).todense())
                similarity = cosine_similarity(query_vector, feedback_vectors)
                nn_similarity = np.max(similarity)
                
                nn_idx = np.argmax(similarity)
                pos_feedback_doc_idx = [idx for idx, feedback_value in self.feedback[feedback_queries[nn_idx]]
                                        if feedback_value == 1.]
                
                counts = Counter(pos_feedback_doc_idx)
                feature_values = {
                    doc_idx: nn_similarity * count / sum(counts.values()) 
                    for doc_idx, count in counts.items()
                }
                return np.array([feature_values.get(doc_idx, 0.) for doc_idx, _ in enumerate(self.docs)])
            
        scorer = Scorer(docs)
        """
        print("SCORE OF DOCUMENTS BEFORE FEEDBACK")
        print(scorer.score(query))

        print("TOP", str(numResults),"DOCUMENTS")
        topDocs = []
        for i in range(numResults):
            #print(scorer.score(query).argmax())
            topDocs.append(docs[scorer.score(query).argmax()])
            del docs[scorer.score(query).argmax()]
            scorer = Scorer(docs)

        for i,topdoc in enumerate(topDocs):
            print("Doc " + str(i+1) + " from Source: " + sources[i] + ",",topdoc)

        print("LEARNING FEEDBACK...")
        scorer.learn_feedback(feedback)

        print("SCORE OF DOCUMENTS AFTER FEEDBACK")
        print(scorer.score(query))
        """
        print("TOP", str(numResults),"DOCUMENTS")
        topDocs = []
        
        for i in range(numResults):
            #print(scorer.score(query).argmax())
            topDocs.append(docs[scorer.score(query).argmax()])
            del docs[scorer.score(query).argmax()]
            scorer = Scorer(docs)

        for i,topdoc in enumerate(topDocs):
            print("Doc " + str(i+1) + " from Source: " + sources[i] + ",",topdoc)
        progress = {"progress": 100, "status": "Finished!", "results": topDocs, "sources": sources}
        print(f"Results: {progress}")  # Optional: Display progress
        return progress["results"]
    else:
        print("Error: no search results have been found")
        return ["None"]
    
def get_progress():
    """Return current progress state."""
    global progress
    return {
        "progress": progress["progress"],
        "status": progress["status"],
        "completed": progress["progress"] >= 100
    }

def get_results():
    """Return final results after completion."""
    global progress
    try:
        results = progress["results"] or ["No results found."]
        return results
    except Exception as e:
        logging.error(f"Error in get_results: {e}")
        return ["Error retrieving results."]
    
def get_sources():
    global progress
    try:
        sources = progress["sources"] or ["No results found."]
        return sources
    except Exception as e:
        logging.error(f"Error in get_sources: {e}")
        return ["Error retrieving sources."]
    

def reset_progress():
    global _progress_lock
    global progress
    # Global progress variable
    _progress_lock = threading.Lock()
    # Reset progress
    with _progress_lock:
        progress = {"progress": 0, "status": "Processing", "results": None}
    print("Progress: " + progress["status"])