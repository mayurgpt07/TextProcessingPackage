import warnings
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser

warnings.filterwarnings("ignore")

class Topic_Modelling:

    def __init__(self, data, column_name, vectorizer_type, required_parameters, is_sklearn = True):
        self.data = data
        self.column_name = column_name
        self.type = vectorizer_type
        self.is_sklearn = is_sklearn
        self.vectorizers = dict()
        self.parameters = required_parameters


    def bow_vectorizer_sklearn(corpus, parameters):
        vectorizer = CountVectorizer()
        vectorizer.fit(corpus)
        vector = vectorizer.transform(corpus)
        self.vectorizers['BOW_Sklearn'] = vectorizer
        return vector, self.vectorizers

    def bow_vectorizer_gensim(corpus, parameters):
        corpus_dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        self.vectorizers['BOW_Gensim'] = corpus_dictionary
        return bow_corpus, self.vectorizers
