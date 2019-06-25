import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from train_data import gen_train_data
from rst_parser import parse_files
from model import mini_batch_linear_model 
from model import neural_network_model
from vocabulary import gen_vocabulary
from preprocess import print_trees_stats

import sys

# Directories variables
WORK_DIR = "." # current dir 
TRAINING_DIR = "..\\dataset\\TRAINING" # directory of the training dataset
DEV_TEST_DIR = "..\\dataset\\DEV" # directory of input dev/test dataset
DEV_TEST_GOLD_DIR = "dev_gold" # dir of the output golden serial trees of dev/test dataset
PRED_OUTDIR = "pred" # directory of the generated predicted serial trees
GLOVE_DIR = "..\\glove" # in which the glove embedding vectors file exists (glove.6B.50d.txt)

def parse_args(argv):
	model_name = "neural"
	baseline = False
	print_stats = False

	if len(argv) < 2:
		return [model_name, baseline, print_stats]

	cmd = "-m <linear|neural> -baseline -stats"

	if len(argv) >= 2:
		i = 1
		while i < len(argv):
			if argv[i] == "-m":
				assert (i + 1) < len(argv), "Model name is missing. Correct cmd: " + cmd 
				model_name = argv[i + 1]
				assert model_name == "linear" or model_name == "neural", \
					"Bad model name: " + argv[i + 1] + " Use linear|neural"
				i += 1
			elif argv[i] == "-baseline":
				baseline = True
			elif argv[i] == "-stats":
				print_stats = True
			else:
				assert False, "bad command line. Correct cmd: " + cmd
			i += 1

	return [model_name, baseline, print_stats]

def train_model(model_name, trees, samples, y_all, vocab, tag_to_ind_map):
	if model_name == "neural":
		model = neural_network_model(trees, samples, vocab, tag_to_ind_map)
	else:
		model = mini_batch_linear_model(trees, samples, y_all, vocab, \
			tag_to_ind_map)

	return model
	
if __name__ == '__main__':
	[model_name, baseline, print_stats] = parse_args(sys.argv)

	print("preprocessing")
	trees = preprocess(WORK_DIR, TRAINING_DIR)
	if print_stats:
		print_trees_stats(trees)

	[vocab, tag_to_ind_map] = gen_vocabulary(trees, WORK_DIR, TRAINING_DIR, GLOVE_DIR)

	model = '' # model data structure
	y_all = '' # for the linear model, y labels are computed dynamically
	if not baseline:
		print("training..")
		[samples, y_all] = gen_train_data(trees, WORK_DIR)
		model = train_model(model_name, trees, samples, y_all, \
			vocab, tag_to_ind_map)

	print("evaluate..")
	dev_trees = preprocess(WORK_DIR, DEV_TEST_DIR, DEV_TEST_GOLD_DIR)

	parse_files(WORK_DIR, model_name, model, dev_trees, vocab, \
		y_all, tag_to_ind_map, baseline, DEV_TEST_DIR, \
		DEV_TEST_GOLD_DIR, PRED_OUTDIR)
