import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,  TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import logging
import json
from sklearn.svm import SVC
import embeddingvectorizer
from sklearn.ensemble import ExtraTreesClassifier
import gensim
import os


path_to_embeddings='/home/anne/tmpanne/fullsample'
path_to_data='/home/anne/tmpanne/AEM_data/'
dataset = 'dataset_vermeer.pkl'
outputpath = '/home/anne/tmpanne/AEM_output/'

#word_embedding_path = '/home/anne/tmpanne/fullsample/'

class classifier_analyzer():
    
    def __init__(self, path_to_data, path_to_embeddings, dataset):
        self.nmodel = 0
        df = pd.read_pickle(path_to_data + dataset)
        logging.info("... loading the data...\n\nthis is length of the dataframe: {}".format(len(df)))
        self.test_size = 0.2
        self.data = df['text']
        self.labels = df['topic']  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=self.test_size, random_state=42)        
        self.basepath = path_to_embeddings
        self.names = ["Passive Agressive", "SGDClassifier" , "SVM", "ET"]
        self.classifiers = [PassiveAggressiveClassifier(), 
                            SGDClassifier(),
                            SVC(),
                            ExtraTreesClassifier() ]
        self.parameters = [
                    {'clf__loss': ('hinge', 'squared_hinge'),
                    'clf__C': (0.01, 0.5, 1.0)   ,
                    'clf__fit_intercept': (True, False) ,
                    'clf__max_iter': (5 ,10 ,15) } ,

                    {'clf__max_iter': (20, 30) ,
                    'clf__alpha': (1e-2, 1e-3, 1e-5),
                    'clf__penalty': ('l2', 'elasticnet')} ,

                    {'clf__C': [1, 10, 100, 1000],
                    'clf__gamma': [0.001, 0.0001],
                    'clf__kernel': ['rbf', 'linear']},
            
                    { "clf__max_features": ['auto', 'sqrt', 'log2'] } ]


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


    def get_vectorizer(self, vectorizer, model):
        logging.info("the vectorizer is: {}".format(vectorizer))
        vec = {}   
        vec['filename'] = vectorizer
        if vectorizer == 'w2v_count':
            s = embeddingvectorizer.EmbeddingCountVectorizer(model['gensimmodel'], 'mean')
        elif vectorizer == 'w2v_tfidf':
            s = embeddingvectorizer.EmbeddingTfidfVectorizer(model['gensimmodel'], 'mean')
        vec['vectorizer'] = s
        print("loaded vectorizer, named: {}".format(vec['filename']))
        yield vec


    def gridsearch_with_classifiers_embeddings(self):
        class_report = []
        results = []
        
        for model in self.get_w2v_model():
            for v in ["w2v_count", "w2v_tfidf"]:
                for vec in self.get_vectorizer(v, model):
                    print("loaded the vectorizer: {}".format(vec['filename'])) 
                    for name, classifier, params in zip(self.names, self.classifiers, self.parameters):
                        my_dict = {}
                        
                        logging.info("Starting gridsearch CV..")
                        logging.info("Classifier name: {}\n classifier:{}\n params{}\n".format(name, classifier, params)) 

                        clf_pipe = Pipeline([ ('vect', vec['vectorizer']), ('clf', classifier), ])

                        gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=2)
                        clf = gs_clf.fit(self.X_train, self.y_train)
                        score = clf.score(self.X_test, self.y_test)

                        logging.info("{} score: {}".format(name, score))
                        logging.info("{} are the best estimators".format(clf.best_estimator_))

                        results_to_dict = classification_report((clf.best_estimator_.predict(self.X_test)), self.y_test, output_dict= True)

                        results_to_dict['classifier'] = name
                        results_to_dict['parameters'] = clf.best_params_
                        results_to_dict['vectorizer'] = vec['filename']
                        results_to_dict['model'] = model['filename']

                        logging.info("Created dictionary with classification report: \n\n{}".format(results_to_dict))
                        class_report.append(results_to_dict)
                        
                        y_hats = clf.predict(self.X_test)
                        results.append({"predicted": y_hats,
                                        "actual" : self.y_test.values  ,
                                        "classifier": name} )
                        
        return class_report, results

    def gridsearch_with_classifiers_baseline(self):
        class_report = []
        results = []
        
        for vec, n in zip([CountVectorizer(), TfidfVectorizer()], ["Count", "TfidF"]):
            
            print("loaded the vectorizer: {}\n\n\{}".format(n, vec)) 
            
            for name, classifier, params in zip(self.names, self.classifiers, self.parameters):
                my_dict = {}

                logging.info("Starting gridsearch CV..")
                logging.info("Classifier name: {}\n classifier:{}\n params{}\n".format(name, classifier, params)) 

                clf_pipe = Pipeline([ ('vect', vec), ('clf', classifier), ])

                gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=2)
                clf = gs_clf.fit(self.X_train, self.y_train)
                score = clf.score(self.X_test, self.y_test)

                logging.info("{} score: {}".format(name, score))
                logging.info("{} are the best estimators".format(clf.best_estimator_))

                results_to_dict = classification_report((clf.best_estimator_.predict(self.X_test)), self.y_test, output_dict= True)

                results_to_dict['classifier'] = name
                results_to_dict['parameters'] = clf.best_params_
                results_to_dict['vectorizer'] = n
                results_to_dict['model'] = "baseline"
                
                logging.info("Created dictionary with classification report: \n\n{}".format(results_to_dict))
                class_report.append(results_to_dict)

                y_hats = clf.predict(self.X_test)
                results.append({"predicted": y_hats,
                                "actual" : self.y_test.values  ,
                                "classifier": name} )
                
        return class_report, results

def get_scores(path_to_data, path_to_embeddings, dataset, outputpath):
    a = classifier_analyzer(path_to_data=path_to_data, path_to_embeddings=path_to_embeddings, dataset=dataset)
    class_report, results = a.gridsearch_with_classifiers_baseline()

    fname_accuracy = '{}baseline_classreport.json'.format(outputpath)
    fname_true_predicted = '{}baseline_true_predicted.json'.format(outputpath)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)

    data = pd.DataFrame.from_dict(results)

    predicted = data.predicted.apply(pd.Series).merge(data, right_index = True, left_index = True) \
        .drop(["predicted"], axis = 1).melt(id_vars = ['classifier'], value_name = "Predicted label")

    actual = data.actual.apply(pd.Series).merge(data, right_index = True, left_index = True) \
        .drop(["predicted"], axis = 1).melt(id_vars = ['classifier'], value_name = "Actual label")

    df = pd.merge(predicted, actual, how = 'inner', left_index = True, right_index = True)
    
    df['Classifier'] = df['classifier_x']
    df = df[df.variable_x != 'actual']
    df = df[['Predicted label', 'Actual label', 'Classifier']]

    df.to_json(fname_true_predicted)

    
if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    get_scores(path_to_embeddings=path_to_embeddings,path_to_data=path_to_data, dataset=dataset, outputpath = outputpath)

#"w2v_count", "w2v_tfidf", "count", "tfidf"

print("DOOOOOOOOOOOOONEEE")
