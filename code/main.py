import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from train_data import gen_train_data
from rst_parser import evaluate
from model import train_model
from vocabulary import gen_vocabulary
from preprocess import print_trees_stats
from defs import *

import sys

def parse_args(argv):
	model_name = "neural"
	baseline = False
	print_stats = False
	k_top = 1

	if len(argv) < 2:
		return [model_name, baseline, print_stats, k_top]

	cmd = "-m <dplp_A_0|dplp_A_I||neural> -baseline -stats -k_top"

	if len(argv) >= 2:
		i = 1
		while i < len(argv):
			if argv[i] == "-m":
				assert (i + 1) < len(argv), "Model name is missing. Correct cmd: " + cmd 
				model_name = argv[i + 1]
				assert model_name in ['dplp_A_0', 'dplp_A_I', 'neural', 'dplp'], \
					"Bad model name: " + argv[i + 1] + " Use neural|dplpdplp_A_0|dplp_A_I|dplp"
				i += 1
			elif argv[i] == "-baseline":
				baseline = True
			elif argv[i] == "-stats":
				print_stats = True
			elif argv[i] == "-k_top":
				k_top = int(argv[i + 1])
				i += 1
			else:
				assert False, "bad command line. Correct cmd: " + cmd
			i += 1

	if baseline and k_top > 1:
		assert False, "-k_top must be 1 when running with -baseline"

	return [model_name, baseline, print_stats, k_top]
	
if __name__ == '__main__':
	[model_name, baseline, print_stats, k_top] = parse_args(sys.argv)

	print("preprocessing")
	trees = preprocess(WORK_DIR, TRAINING_DIR)
	if print_stats:
		print_trees_stats(trees)

	[vocab, tag_to_ind_map] = gen_vocabulary(trees, WORK_DIR, TRAINING_DIR, GLOVE_DIR)

	model = '' # model data structure
	if not baseline:
		print("training...")
		[samples, y_all] = gen_train_data(trees, WORK_DIR)
		model = train_model(model_name, trees, samples, vocab, tag_to_ind_map)

	print("evaluate..")
	evaluate(model_name, model, vocab, tag_to_ind_map, baseline, k_top)
