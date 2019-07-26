import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from gensim.models.word2vec import Word2Vec
import gensim
from sklearn.ensemble import ExtraTreesClassifier
import embeddingvectorizer
from sklearn.metrics import precision_recall_fscore_support as score
import json
import os

class word2vec_analyzer():
    '''This class tests the efficacy of Word2Vec models in downstream tasks.'''

    def __init__(self, path_to_data, path_to_embeddings, vectorizer, sample_size):
        self.nmodel = 0
        self.vectorizer = vectorizer
        if path_to_data.data_path.split('/')[-1] == 'dataset_vermeer.pkl':
            self.train_sizes = [10, 40, 160, 640, 1000, 2500, 3486]
        elif path_to_data.data_path.split('/')[-1] == 'dataset_burscher.pkl':
            self.train_sizes = [10, 40, 160, 640, 1000, 2500, 5000, 7500, 10000, 12559]
        df = pd.read_pickle(path_to_data)
        self.data = df['text']
        self.labels = df['topic']
        self.samplesize = sample_size
        self.basepath = path_to_embeddings

    def get_w2v_model(self):
        '''yields a dict with one item. key is the filename, value the gensim model'''

        filenames = [e for e in os.listdir(self.basepath) if not e.startswith('.')]

        for fname in filenames:
            model = {}
            path = os.path.join(self.basepath, fname)
            print("\nLoading gensim model")

            if fname.startswith('w2v'):
                mod = gensim.models.Word2Vec.load(path)
            else:
                mod = gensim.models.KeyedVectors.load_word2vec_format(path)

            model['gensimmodel'] = dict(zip(mod.wv.index2word, mod.wv.vectors))
            model['filename'] = fname
            self.nmodel +=1
            print("loaded gensim model nr {}, named: {}".format(self.nmodel, model['filename']))
            yield model

    def benchmark(self, model, X, y, n):
        test_size = 1 - (n / float(len(y)))
        print("test size: {}".format(test_size))
        #precision_ , recall_  , f1score_ , scores = ([] for i in range(4)) # create 4 empty lists
        scores = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        #scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
        precision,recall,fscore,support=score(y_test, preds, average='weighted')
        #recall_.append(recall)
        #precision_.append(precision)
        #f1score_.append(fscore)
        print("precision: {}, recall: {};, f1socre: {}, mean (accuracy) scores: {}".format(precision, recall, fscore, np.mean(scores)))
        return np.mean(scores), recall, precision, fscore

    def get_baseline_results(self):
        results = []
        if self.vectorizer == "Tfidf":
            print(">>>> defining pipes for baseline model with {} vectorizer".format( self.vectorizer))

            svm = Pipeline([("vect", TfidfVectorizer()),
                            ("svm", SGDClassifier(loss='hinge', penalty='elasticnet', tol=1e-4, alpha=1e-6, max_iter=5000, random_state=42))
                            ])


            ET = Pipeline([
                ("vect", TfidfVectorizer()),
                ("ExtraTrees", ExtraTreesClassifier(n_estimators=200))
                ])


        else:
            print(">>>> defining pipes for baseline model with {} vectorizer".format( self.vectorizer))
            svm = Pipeline([("vect", CountVectorizer()),
                            ("svm", SGDClassifier(loss='hinge', penalty='elasticnet', tol=1e-4, alpha=1e-6, max_iter=5000, random_state=42))
                            ])

            ET = Pipeline([
                ("vect", CountVectorizer()),
                ("ExtraTrees", ExtraTreesClassifier(n_estimators=200))
                ])

        all_models = [ ( "svm" , svm ) , ("ET", ET), ]

        results = []
        for name, model in all_models:
            for n in self.train_sizes:
                print(name)
                results.append({'classifier': name,
                                'model' : "baseline" ,
                                'accuracy': self.benchmark(model, self.data, self.labels, n)[0],
                                'recall': self.benchmark(model, self.data, self.labels, n)[1],
                                'precision': self.benchmark(model, self.data, self.labels, n)[2],
                                'f1score': self.benchmark(model, self.data, self.labels, n)[3],
                                'train_size': n})

        return results


    def get_scores_wv2(self):

        results = []

        for model in self.get_w2v_model():

            if self.vectorizer == "Tfidf":
                print(">>>> defining pipes for model {} with {} vectorizer".format(model['filename'], self.vectorizer))

                w2v_svm = Pipeline([
                ("word2vec TfidF vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(model['gensimmodel'])),
                ("svm", SGDClassifier(loss='hinge', tol=1e-4, alpha=1e-6, max_iter=5000, random_state=42, penalty='elasticnet'))
                ])

                w2v_ET = Pipeline([
                ("word2vec tfidf vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(model['gensimmodel'])),
                ("ExtraTrees", ExtraTreesClassifier(n_estimators=200))
                ])

            else:
                print(">>>> defining pipes for model {} with {} vectorizer".format(model['filename'], self.vectorizer))

                w2v_svm = Pipeline([
                ("word2vec TfidF vectorizer", embeddingvectorizer.EmbeddingCountVectorizer(model['gensimmodel'])),
                ("svm", SGDClassifier(loss='hinge', tol=1e-4, alpha=1e-6, max_iter=5000, random_state=42, penalty='elasticnet'))
                ])

                w2v_ET = Pipeline([
                ("word2vec tfidf vectorizer", embeddingvectorizer.EmbeddingCountVectorizer(model['gensimmodel'])),
                ("ExtraTrees", ExtraTreesClassifier(n_estimators=200))
                ])

            all_models = [ ( "w2v_svm" , w2v_svm ) , ("w2v_ET ", w2v_ET ), ]

            table = []
            for name, m in all_models:
                for n in self.train_sizes:
                    print(name)
                    table.append({'classifier': name,
                                    'model' : model['filename'] ,
                                    'accuracy': self.benchmark(m, self.data, self.labels, n)[0],
                                    'recall': self.benchmark(m, self.data, self.labels, n)[1],
                                    'precision': self.benchmark(m, self.data, self.labels, n)[2],
                                    'f1score': self.benchmark(m, self.data, self.labels, n)[3],
                                    'train_size': n})
            results.append(table)
        return results


    def get_final(self):
        results_wv2 = self.get_scores_wv2()
        results_baseline = self.get_baseline_results()
        return results_baseline, results_wv2
