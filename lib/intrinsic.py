class word2vec_analyzer():
    '''This class tests the efficacy of Word2Vec models in downstream tasks.'''

    def __init__(self, path_to_data, path_to_embeddings, vectorizer, sample_size):
        self.nmodel = 0
        self.vectorizer = vectorizer
        self.train_sizes = [10, 40, 160, 640, 1000, 2500]
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
