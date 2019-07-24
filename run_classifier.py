from lib import classification
import logging
import argparse
import json
import pandas as pd

def main(args):
	analyzer = classification.word2vec_analyzer(path_to_data = args.data_path, path_to_embeddings = args.word_embedding_path, vectorizer = args.type_vectorizer, sample_size = args.word_embedding_sample_size)
	my_results = analyzer.get_final()
	print(my_results)

	flat_list = [item for sublist in my_results[1] for item in sublist]
	w2v = pd.DataFrame.from_dict(flat_list)
	baseline = pd.DataFrame.from_dict(my_results[0])
	df = pd.concat([w2v, baseline])
	print('Created dataframe')
	print(df)

	df.to_pickle('{}_training_size_{}_{}.pkl'.format(args.output, args.word_embedding_sample_size, args.data_path.split('/')[-2]))


	with open('{}_training_size_{}_{}.json'.format(args.output, args.word_embedding_sample_size, args.data_path.split('/')[-2]),mode='w') as fo:
		fo.write('[')
		for result in my_results:
			fo.write(json.dumps(result))
			fo.write(',\n')
		fo.write('[]]')
		print("\n\n\nSave results\n\n\n")


if __name__ == '__main__':

	logger = logging.getLogger()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
	logging.root.setLevel(level=logging.INFO)

	parser = argparse.ArgumentParser(description='Compute accuracy score of pretrained word embedding models')
	parser.add_argument('--word_embedding_sample_size', type=str, required=False, default = 'large', help='Size of sample of pretrained word embedding (small or large)')
	parser.add_argument('--type_vectorizer', type=str, required=False, default = 'Tfidf', help='Type of vectorizer (Tfidf or count)')
	parser.add_argument('--word_embedding_path', type=str, required=True, help='Path of pretrained word embedding.')
	parser.add_argument('--data_path', type=str, required=False, default='data/dataset_vermeer.pkl', help='Path of dataset with annotated data to be classified')
	parser.add_argument('--output', type=str, required=False, default='output/output', help='Path of output file (CSV formatted classification scores)')
	args = parser.parse_args()

	print('Arguments:')
	print('word_embedding_sample_size:', args.word_embedding_sample_size)
	print('word_embedding_path:', args.word_embedding_path)
	print('data_path:', args.data_path)
	print('output.path:', args.output)
	print()

	main(args)
