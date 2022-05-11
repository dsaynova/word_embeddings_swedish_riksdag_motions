import pickle
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from collections import defaultdict
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_span', choices = ['1988-2009', '1988-2020', '2010-2020'], required=True)
    parser.add_argument('--party', choices = ['s', 'm'], required=True)
    parser.add_argument('--pre_trained_model', type = str, required=True)
    parser.add_argument('--epochs', type = int, required=True)
    #USED IN EXPERIMENTS: 
    #'1988-2009' : 50; '1988-2020' : 50; '2010-2020' : 100
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

    #FINE-TUNE MODEL
    for iter_num in range(10):
        print()

        #LOAD PRE-TRAINED MODEL
        model = Word2Vec.load(args.pre_trained_model)

        #BUILD VOCAB
        model.build_vocab(training_examples, update=True)
        
        #RESAMPLE DATA FOR BOOTSTRAPPING
        training_examples_a = sklearn.utils.resample(training_examples, replace=True, n_samples=len(training_examples))

        #TRAIN MODEL
        model.train(training_examples_a, total_examples=len(training_examples_a), epochs=args.epochs, compute_loss=True, callbacks=[callback()])

        #EXTRAXT AND SAVE LEARNED EMBEDDINGS
        res = model.wv
        if args.output_folder:
            out = args.output_folder
        else: out = ''
        res.save(out+"riksdag_"+args.time_span.replace('-','_')+"_"+args.party+"_"+str(iter_num)+'.kv')


if __name__ == "__main__":
    main()

