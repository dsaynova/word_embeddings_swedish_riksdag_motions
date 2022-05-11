import pickle
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import operator
import sklearn
import argparse


#Callback to print loss after each epoch
#code from: https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
#fix needed related to: https://github.com/RaRe-Technologies/gensim/pull/2135
class callback(CallbackAny2Vec):
    
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.epoch += 1
        #print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now), end='\r')


#LOAD VECTORS FROM EXISTING MODEL, IF NOT IN MODEL - INIT TO RANDOM
def load_vectors(token2id, path):
    embed_shape = (len(token2id), 100)

    vectors = np.random.uniform(low=-1.0, high=1.0, size=embed_shape)
    with open(path, encoding="utf8", errors='ignore') as f:
        for line in f:
            token, *vector = line.rstrip().split(' ')
            token = str.lower(token)
            if len(line) <= 100:
                continue
            vectors[token2id[token]] = np.array(vector, 'f')

    return vectors

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_span', choices = ['1988-2009', '1988-2020', '2010-2020'], required=True)
    parser.add_argument('--party', choices = ['s', 'm'], required=True)
    parser.add_argument('--pre_trained_model', type = str, required=True)
    parser.add_argument('--epochs', type = int, required=True)
    #USED IN EXPERIMENTS: 
    #'1988-2009' : 100; '1988-2020' : 50; '2010-2020' : 150
    parser.add_argument('--output_folder', type=str, required=False)
    args = parser.parse_args()

    #READ DATA
    training_examples = []
    data_dict = defaultdict()
    if args.time_span in ['1988-2009', '1988-2020']:
        with open('data_1988_2009.pkl', 'rb') as f:
            data_dict.update(pickle.load(f))

    if args.time_span in ['1988-2020', '2010-2020']:
        with open('data_2010_2020.pkl', 'rb') as f:
            data_dict.update(pickle.load(f))

    for key in data_dict.keys():
        if data_dict[key][1] == args.party:
            training_examples.append(data_dict[key][2].strip().split(" "))


    #CREATE FREQ AND ID MAPPING FOR TOKENS
    #code from: https://stackoverflow.com/questions/56166089/wor2vec-fine-tuning
    token2id = {}
    vocab_freq_dict = {}

    # Populating vocab_freq_dict and token2id from data.
    id_ = 0
    for line in training_examples:
        for word in line:
            if word not in vocab_freq_dict:
                vocab_freq_dict.update({word:0})
            vocab_freq_dict[word] += 1
            if word not in token2id:
                token2id.update({word:id_})
                id_ += 1

    # Populating vocab_freq_dict and token2id from external vocab.
    max_id = max(token2id.items(), key=operator.itemgetter(1))[0]
    max_token_id = token2id[max_id]
    with open(args.pre_trained_model, encoding="utf8", errors='ignore') as f:
        for o in f:
            token, *vector = o.split(' ')
            token = str.lower(token)
            if len(o) <= 100:
                continue
            if token not in token2id:
                max_token_id += 1
                token2id.update({token:max_token_id})
                vocab_freq_dict.update({token:1})

    #FINE-TUNE MODEL
    for iter_num in range(10):
        print()
        #INITIALIZE VECTORS
        vectors = load_vectors(token2id, args.pre_trained_model)
        vec = KeyedVectors(100)
        vec.add(list(token2id.keys()), vectors, replace=True)

        #INITIALIZE MODEL
        params = dict(min_count=1,workers=2,iter=args.epochs,size=100)
        model = Word2Vec(**params)

        #BUILD VOCAB
        model.build_vocab_from_freq(vocab_freq_dict)

        #INITIALIZE MODEL WEIGHTS
        idxmap = np.array([token2id[w] for w in model.wv.index2entity])
        #weights between input layer and hidden layer
        model.wv.vectors[:] = vec.vectors[idxmap]
        #weights between hidden layer and output layer
        model.trainables.syn1neg[:] = vec.vectors[idxmap]

        #RESAMPLE DATA FOR BOOTSTRAPPING
        training_examples_a = sklearn.utils.resample(training_examples, replace=True, n_samples=len(training_examples))

        #TRAIN MODEL
        model.train(training_examples_a, total_examples=len(training_examples_a), epochs=model.epochs, compute_loss=True, callbacks=[callback()])
        
        #EXTRAXT AND SAVE LEARNED EMBEDDINGS
        res = model.wv
        if args.output_folder:
            out = args.output_folder
        else: out = ''
        res.save(out+"nlpl_"+args.time_span.replace('-','_')+"_"+args.party+"_"+str(iter_num)+'.kv')



if __name__ == "__main__":
    main()

