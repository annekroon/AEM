import nltk
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


path_to_models = '/Users/anne/repos/embedding_models/'

path_to_data ='../data/'
df = pd.read_pickle(path_to_data + "data_geannoteerd.pkl")

text =df['text']
df['text_stop']= df['text'].str.lower()
stop = set(stopwords.words('dutch'))
df['text_stop']= df['text_stop'].str.split()


