from gensim.models import KeyedVectors
from collections import defaultdict
import numpy as np
import pickle
import glob
import os
import argparse


def aggregate_list(models, term, topn, vocab):
    term_dict = defaultdict(list)
    for model in models:
        wv = KeyedVectors.load(model)
        current = wv.most_similar(term, topn = len(wv.vocab))
        for k,v in current:
            if k not in vocab: continue
            else:        
                term_dict[k].append(v)

    final_dict = defaultdict()
    for k in term_dict.keys():
        final_dict[k] = [np.mean(term_dict[k]), np.std(term_dict[k])]
    s = sorted(final_dict.items(), key=lambda item: item[1][0])
    s.reverse()
    return s[0:topn]

def print_to_file(time_span, party, pre_train, model_folder):
    #READ DATA
    vocab = set()
    term_count = defaultdict(int)
    doc_count = defaultdict(set)
    data_dict = defaultdict()

    if time_span in ['1988-2009', '1988-2020']:
        with open('data_1988_2009.pkl', 'rb') as f:
            data_dict.update(pickle.load(f))

    if time_span in ['1988-2020', '2010-2020']:
        with open('data_2010_2020.pkl', 'rb') as f:
            data_dict.update(pickle.load(f))

    for key in data_dict.keys():
        if data_dict[key][1] == party:
            for token in data_dict[key][2].strip().split(" "):
                term_count[token] += 1
                doc_count[token].update([key])
                vocab.add(token)



    terms = ['droger', 'skatt', 'säkerhet', 'trygghet', 'brott',\
                'brottslighet', 'kriminalitet', 'jämlikhet', 'solidaritet', 'rättvisa']                

    models = glob.glob(os.path.join(model_folder,'**',pre_train)+"_"+\
        time_span.replace('-','_')+"_"+party+"_*"+'.kv', recursive=True)

    f_name = pre_train+'_'+time_span+'_'+party+'.txt'
    with open(f_name, 'w') as f:
        for t in terms:
            f.write("%s\n" % (t))
            f.write("term|score|std|term frequency|document frequency\n")
            for a,b in aggregate_list(models, t, 20, vocab):
                f.write("%s|%f|%f|%d|%d\n" % (a,b[0],b[1],term_count[a],len(doc_count[a])))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--time_span', nargs = '+', choices = ['1988-2009', '1988-2020', '2010-2020'])
    parser.add_argument('--party', nargs = '+', choices = ['s', 'm'])
    parser.add_argument('--pre_train', nargs = '+', choices = ['nlpl', 'riksdag'])
    args = parser.parse_args()

    for p in args.party:
        for ts in args.time_span:
            for pt in args.pre_train:
                print(p,ts,pt)
                print_to_file(ts, p, pt, args.model_folder)



if __name__ == "__main__":
    main()
