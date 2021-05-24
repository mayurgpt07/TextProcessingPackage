import warnings
from CustomException import CustomException
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class Topic_Modelling:

    def __init__(self, data, column_name, vectorizer_type = 'bow', topic_modelling_type = 'lda', vectorizer_parameters = {}, topic_model_parameters = {}, num_topics = 10, show_visualization = True, save_fig = False, fig_title = 'Topics in the data', is_sklearn = True):
        self.data = data
        self.column_name = column_name
        self.type = vectorizer_type
        self.topic_model = topic_modelling_type
        self.is_sklearn = is_sklearn
        self.vectorizers = dict()
        self.topic_models_used = dict()
        self.show_visualization = show_visualization
        self.save_fig = save_fig
        self.num_topics = num_topics
        self.fig_name = fig_title

        # Multiple parameter problem, setting parameters for LDA gensim and TFIDF gensim
        if type(vectorizer_parameters) is dict:
            self.parameters = vectorizer_parameters
        else:
            raise CustomException('Type Mismatch: The parameters should be of type dicts')

        if type(topic_model_parameters) is dict:
            self.model_parameters = topic_model_parameters
        else:
            raise CustomException('Type Mismatch: The parameters should be of type dicts')
                
    
    def plot_top_words(self, model, feature_names, n_top_words, title, save_fig):
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
        if save_fig:
            plt.savefig('TopicModel.png')
        plt.show()

    def bow_vectorizer_sklearn(self, corpus, parameters):
        vectorizer = CountVectorizer()
        if parameters:
            try:
                vectorizer.set_params(**parameters)
            except:
                raise CustomException('Parameter Error: Check the name and values of parameter for CountVectorizer')
        print(vectorizer)
        vectorizer.fit(corpus)
        vector = vectorizer.transform(corpus)
        self.vectorizers['BOW_Sklearn'] = vectorizer
        return vector

    def vectorizer_gensim(self, corpus, parameters, vectorizer_type):
        corpus_dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        self.vectorizers['BOW_Gensim'] = bow_corpus
        if vectorizer_type == 'tfidf':
            tfidf = models.TfidfModel(bow_corpus, id2word=corpus_dictionary)
            tfidf_corpus = tfidf[bow_corpus]
            self.vectorizers['TFIDF_Gensim'] = tfidf_corpus
            return tfidf_corpus
        elif vectorizer_type == 'bow':
            return bow_corpus
        else:
            raise CustomException("Unknow type of vectorizer")


    def tfidf_vectorizer_sklearn(self, corpus, parameters):
        vectorizer = TfidfVectorizer()
        if parameters:
            try:
                vectorizer.set_params(**parameters)
            except:
                raise CustomException('Parameter Error: Check the name and values of parameter for TF-IDF Vectorizer')
        print(vectorizer)
        vectorizer.fit(corpus)
        vector = vectorizer.transform(corpus)
        self.vectorizers['TFIDF_Sklearn'] = vectorizer
        return vector

    def tfidf_vectorizer_gensim(self, corpus, parameters):
        corpus_dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        self.vectorizers['TFIDF_Gensim'] = corpus_dictionary
        return bow_corpus

    def lda_topic_modeling(self, count_features, parameters, feature_names, save_fig, show_visualization = True):
        lda = LatentDirichletAllocation()
        if parameters:
            try:
                lda.set_params(**parameters)
            except:
                raise CustomException('Parameter Error: Check the name and values of parameter for LatentDirichiletAllocation')
        print(lda)
        lda.fit(count_features)
        self.topic_models_used['LDA_Sklearn'] = lda
        output = lda.fit_transform(count_features)
        feature_names = feature_names
        if show_visualization:
            # Call the visualizations
            self.plot_top_words(lda, feature_names, self.num_topics, self.fig_name, save_fig)
        return output

    def nmf_topic_modeling(self, features, parameters, feature_names, save_fig, show_visualization = True):
        nmf = NMF()
        if parameters:
            if 'n_components' not in parameters.keys():
                parameters['n_components'] = 10
            try:
                nmf.set_params(**parameters)
            except:
                raise CustomException('Parameter Error: Check the name and values of parameter for Non-Negative Matrix Factorization')
        else:
            if 'n_components' not in parameters.keys():
                parameters['n_components'] = 10
            try:
                nmf.set_params(**parameters)
            except:
                raise CustomException('Parameter Error: Check the name and values of parameter for Non-Negative Matrix Factorization')

        print(nmf)
        nmf.fit(features)
        self.topic_models_used['NMF_Sklearn'] = nmf
        output = nmf.fit_transform(features)
        feature_names = feature_names
        if show_visualization:
            self.plot_top_words(nmf, feature_names, self.num_topics, self.fig_name, save_fig)
        return output

    def return_models(self, return_vectorizer = False, return_topic_model = True):
        if return_vectorizer == True and return_topic_model == True:
            vectorizers_and_models = {}
            vectorizers_and_models['vectorizers'] = self.vectorizers
            vectorizers_and_models['topic_model'] = self.topic_models_used
            return vectorizers_and_models
        elif return_vectorizer == True and return_topic_model == False:
            return self.vectorizers
        elif return_vectorizer == False and return_topic_model == True:
            return self.topic_models_used
        else:
            raise CustomException('Error: Choose either or both models to return')
    

    def fit_transform(self):
        corpus = self.data[self.column_name].to_list()
        # print(corpus)
        if self.is_sklearn:
            if self.type == "bow":
                vector = self.bow_vectorizer_sklearn(corpus, self.parameters)
                feature_names = self.vectorizers['BOW_Sklearn'].get_feature_names()
                if self.topic_model == "lda":
                    output = self.lda_topic_modeling(vector, self.model_parameters, feature_names, self.save_fig, show_visualization = self.show_visualization)
                    return output
                elif self.topic_model == "nmf":
                    output = self.nmf_topic_modeling(vector, self.model_parameters, feature_names, self.save_fig, show_visualization = self.show_visualization)
                    return output
                else: 
                    raise CustomException('Unknown type of Topic model')
            elif self.type == "tfidf":
                vector = self.tfidf_vectorizer_sklearn(corpus, self.parameters)
                feature_names = self.vectorizers['TFIDF_Sklearn'].get_feature_names()
                # print(feature_names)
                if self.topic_model == "lda":
                    raise CustomException('TF-IDF features cannot be used with LDA')
                elif self.topic_model == "nmf":
                    output = self.nmf_topic_modeling(vector, self.model_parameters, feature_names, self.save_fig, show_visualization = self.show_visualization)
                    return output
                else: 
                    raise CustomException('Unknown type of Topic model')

            else:
                raise CustomException("Unknow type of vectorizer")
        else:
            vector = self.vectorizer_gensim(corpus, parameters, self.type)
