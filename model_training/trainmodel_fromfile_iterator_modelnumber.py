import gensim
import logging
import re
import itertools

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')
PATH = "/home/anne/tmpanne/AEM_small_sample/"
FILENAME = "uniekezinnen_2000-01-01_2017-12-31"
# n sentences uniekezinnen_2000-01-01_2017-12-31: 4289076/4289076

NR_MODEL = 0

def get_parameters(model_number):
    nr = int(model_number)

    window = [5, 10, 47615]
    negative = [5, 15]
    size = [100, 300]

    w2v_parameters = []

    for w, n, s in list(itertools.product(window, negative, size)):
        my_dict = {}
        my_dict['window'] = w
        my_dict['negative'] = n
        my_dict['size'] = s
        w2v_parameters.append(my_dict)

    return w2v_parameters[nr]

parameters = get_parameters(NR_MODEL)
print("set parameters:", parameters)

def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self, fromdate,todate):
        self.fromdate = fromdate
        self.todate = todate
        self.sentences = gensim.models.word2vec.PathLineSentences(PATH + FILENAME)
        self.w2v_params = get_parameters(NR_MODEL)
        print("estimating model with the following parameter settings: {}".format(self.w2v_params))

        self.model = gensim.models.Word2Vec(**self.w2v_params)
        self.model.build_vocab(self.sentences)
        print('Build Word2Vec vocabulary for Model {}'.format(NR_MODEL))
        self.model.train(self.sentences,total_examples=self.model.corpus_count, epochs=self.model.iter)
        print('Estimated Word2Vec model')

def train_and_save(fromdate,todate):
    filename = "{}w2v_model_nr_{}_window_{}_size_{}_negsample_{}".format(PATH, NR_MODEL, parameters['window'], parameters['size'],parameters['negative'] )

    casus = train_model(fromdate,todate)

    with open(filename, mode='wb') as fo:
        casus.model.save(fo)
    print('Saved model')
    print("reopen it with m = gensim.models.word2vec.load('{}')".format(filename))
    del(casus)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save(fromdate = "2000-01-01", todate = "2017-12-31")
