import gensim
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    data = gensim.models.word2vec.LineSentence(args.data_file)
    model_sv = gensim.models.Word2Vec(sentences=data, min_count=1,workers=2,iter=5,size=100)
    model_sv.save(args.output_file+".model")

if __name__ == "__main__":
    main()
