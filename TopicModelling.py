import warnings
from CustomException import CustomException
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser

warnings.filterwarnings("ignore")

class Topic_Modelling:

    def __init__(self, data, column_name, vectorizer_type, required_parameters, show_visualization = True, save_fig = False, is_sklearn = True):
        self.data = data
        self.column_name = column_name
        self.type = vectorizer_type
        self.is_sklearn = is_sklearn
        self.vectorizers = dict()
        self.topic_models = dict()
        # Multiple parameter problem, setting parameters for LDA gensim and TFIDF gensim
        if type(required_parameters) is dict:
            self.parameters = required_parameters
        else:
            raise CustomException('Type Mismatch: The parameters should be of type dicts')


    def bow_vectorizer_sklearn(self, corpus, parameters):
        vectorizer = CountVectorizer()
        try:
            vectorizer.set_params(parameters)
        except:
            raise CustomException('Parameter Error: Check the name and values of parameter for CountVectorizer')
        vectorizer.fit(corpus)
        vector = vectorizer.transform(corpus)
        self.vectorizers['BOW_Sklearn'] = vectorizer
        return vector, self.vectorizers

    def vectorizer_gensim(self, corpus, parameters, vectorizer_type):
        corpus_dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        self.vectorizers['BOW_Gensim'] = bow_corpus
        if vectorizer_type == 'tfidf':
            tfidf = models.TfidfModel(bow_corpus, id2word=corpus_dictionary)
            tfidf_corpus = tfidf[bow_corpus]
            self.vectorizers['TFIDF_Gensim'] = tfidf_corpus
            return tfidf_corpus, self.vectorizers
        else:
            return bow_corpus, self.vectorizers

    def tfidf_vectorizer_sklearn(self, corpus, parameters):
        vectorizer = TfidfVectorizer()
        try:
            vectorizer.set_params(parameters)
        else:
            raise CustomException('Parameter Error: Check the name and values of parameter for TF-IDF Vectorizer')
        vectorizer.fit(corpus)
        vector = vectorizer.transform(corpus)
        self.vectorizers['TFIDF_Sklearn'] = vectorizer
        return vector, self.vectorizers

    def tfidf_vectorizer_gensim(self, corpus, parameters):
        corpus_dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        self.vectorizers['TFIDF_Gensim'] = corpus_dictionary
        return bow_corpus, self.vectorizers

    def lda_topic_modeling(self, count_features, parameters, show_visualization = True):
        lda = LatentDirichletAllocation()
        try:
            lda.set_params(parameters)
        raise:
            CustomException('Parameter Error: Check the name and values of parameter for LatentDirichiletAllocation')
        lda.fit(count_features)
        self.topic_models['LDA_Sklearn'] = lda
        output = lda.fit_transform(count_features)
        if show_visualization:
            if save_fig:
                plt.savefig()
            plt.show()
            # Call the visualizations
        return output, self.topic_models

    
    def plot_top_words(model, feature_names, n_top_words, title):
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx +1}',
                        fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

