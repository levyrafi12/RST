import re
import copy
import filecmp
import glob
import nltk
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from collections import defaultdict
import os

from utils import map_to_cluster
from relations_inventory import build_parser_action_to_ind_mapping
# from main import SEP

SEP = "/"

# debugging 
print_sents = False
sents_dir = "sents"

class Node(object):
	def __init__(self):
		self._nuclearity = '' # 'Nucleus' | 'Satellite'
		self._relation = ''
		self._childs = []
		self._type = '' # 'span' | 'leaf' | 'Root'
		self._span = [0, 0]
		self._text = ''

	def copy(self):
		to = Node()
		to._nuclearity = self._nuclearity
		to._relation = self._relation
		to._childs = copy.copy(self._childs)
		to._span = copy.copy(self._span)
		to._text = self._text
		to._type = self._type
		return to

	def print_info(self):
		node_type = self._type
		beg = self._span[0]
		end = self._span[1]
		nuc = self._nuclearity
		rel = self._relation
		text = self._text
		print("node: type= {} span = {},{} nuc={} rel={} text={}".\
			format(node_type, beg, end, nuc, rel, text))

class TreeInfo(object):
	def __init__(self):
		self._fname = '' # file name
		self._root = ''
		self._EDUS_table = ['']
		self._sents = ['']
		self._edu_to_sent_ind = ['']
		self._edu_word_tag_table = [['']]

def preprocess(path, dis_files_dir, ser_files_dir='', bin_files_dir=''):
	build_parser_action_to_ind_mapping()

	[trees, max_edus] = binarize_files(path, dis_files_dir, bin_files_dir)

	if ser_files_dir != '':
		print_serial_files(path, trees, ser_files_dir)

	gen_sentences(trees, path, dis_files_dir)

	# statistics (debugging)
	num_edus = 0
	match_edus = 0

	for tree in trees:
		sent_ind = 1
		n_sents = len(tree._sents)
		fn = build_infile_name(tree._fname, path, dis_files_dir, ["out.edus", "edus"])
		with open(fn) as fh:
			for edu in fh:
				edu = edu.strip()
				edu_tokenized = tokenize.word_tokenize(edu)
				tree._edu_word_tag_table.append(nltk.pos_tag(edu_tokenized))
				tree._EDUS_table.append(edu)
				if not edu in tree._sents[sent_ind]:
					sent_ind += 1
				tree._edu_to_sent_ind.append(sent_ind)
				if edu in tree._sents[sent_ind]:
					match_edus += 1
				num_edus += 1
			# assert(sent_ind < n_sents)

	print("num match between edu and a sentence {} , num edus {} , {}%".\
		format(match_edus, num_edus, match_edus / num_edus * 100.0))

	return [trees, max_edus]

def binarize_files(base_path, dis_files_dir, bin_files_dir):
	trees = []
	max_edus = 0
	path = base_path
	path += SEP
	path += dis_files_dir

	assert os.path.isdir(path), \
		"Path to dataset does not exist: " + dis_files_dir

	path += SEP + "*.dis"
	for fn in glob.glob(path):
		check_path_separator(fn, dis_files_dir)
		tree = binarize_file(fn, bin_files_dir)
		trees.append(tree)
		if tree._root._span[1] > max_edus:
			max_edus = tree._root._span[1]
	return [trees, max_edus]

# return the root of the binarized file

def binarize_file(infn, bin_files_dir):
	stack = []
	with open(infn, "r") as ifh: # .dis file
		lines = ifh.readlines()
		root = build_tree(lines[::-1], stack)

	binarize_tree(root)

	if bin_files_dir != '':
		outfn = infn.split(SEP)[0]
		outfn += SEP
		outfn += bin_files_dir
		outfn += SEP
		outfn += extract_base_name_file(infn)
		outfn += ".out.dis"
		with open(outfn, "w") as ofh:
			print_dis_file(ofh, root, 0)

	tree_info = TreeInfo()
	tree_info._root = root
	tree_info._fname = extract_base_name_file(infn)
	return tree_info

def extract_base_name_file(fn):
	base_name = fn.split(SEP)[-1]
	base_name = base_name.split('.')[0]
	return base_name

# lines are the content of .dis" file

def build_tree(lines, stack):
	line = lines.pop(-1)
	line = line.strip()

	# print("{}".format(line))
 
	node = Node()

	# ( Root (span 1 54)
	m = re.match("\( Root \(span (\d+) (\d+)\)", line)
	if m:
		tokens = m.groups()
		node._type = "Root"
		node._span = [int(tokens[0]), int(tokens[1])]
		stack.append(node)
		return build_tree_childs_iter(lines, stack)

	# ( Nucleus (span 1 34) (rel2par Topic-Drift)
	line.replace("\\TT_ERR", '')
	m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par ([\w-]+)\)", line)
	if m:
		tokens = m.groups()
		node._nuclearity = tokens[0]
		node._type = "span"
		node._span = [int(tokens[1]), int(tokens[2])]
		node._relation = tokens[3]
		stack.append(node)
		return build_tree_childs_iter(lines, stack)

	# ( Satellite (leaf 3) (rel2par attribution) (text _!Southern Co. 's Gulf Power Co. unit_!) )
	m = re.match("\( (\w+) \(leaf (\d+)\) \(rel2par ([\w-]+)\) \(text (.+)", line)
	tokens = m.groups()
	node._type = "leaf"
	node._nuclearity = tokens[0]
	node._span = [int(tokens[1]), int(tokens[1])] 
	node._relation = tokens[2]
	text = tokens[3]
	text = text[2:]
	text = text[:-5]
	node._text = text
	# node.print_info()
	return node
	
def build_tree_childs_iter(lines, stack):
	# stack[-1].print_info()

	while True:
		line = lines[-1]
		line.strip()
		words = line.split()
		if words[0] == ")":
			lines.pop(-1)
			break

		node = build_tree(lines, stack)
		stack[-1]._childs.append(node)
	return stack.pop(-1)

def binarize_tree(node):
	if node._childs == []:
		return

	if len(node._childs) > 2:
		stack = []
		for child in node._childs:
			stack.append(child)

		node._childs = []
		while len(stack) > 2:
			# print("deree > 2")
			r = stack.pop(-1)
			l = stack.pop(-1)

			t = l.copy()
			t._childs = [l, r]
			t._span = [l._span[0], r._span[1]]
			t._type = "span"
			stack.append(t)
		r = stack.pop(-1)
		l = stack.pop(-1)
		node._childs = [l, r]
	else:
		l = node._childs[0]
		r = node._childs[1]

	binarize_tree(l)
	binarize_tree(r)

# print tree in .dis format (called after binarization)

def print_dis_file(ofh, node, level):
	nuc = node._nuclearity
	rel = node._relation
	beg = node._span[0]
	end = node._span[1]
	if node._type == "leaf":
		# Nucleus (leaf 1) (rel2par span) (text _!Wall Street is just about ready to line_!) )
		print_spaces(ofh, level)
		text = node._text
		ofh.write("( {} (leaf {}) (rel2par {}) (text _!{}_!) )\n".format(nuc, beg, rel, text))
	else:
		if node._type == "Root":
			# ( Root (span 1 91)
			ofh.write("( Root (span {} {})\n".format(beg, end))
		else:
			# ( Nucleus (span 1 69) (rel2par Contrast)
			print_spaces(ofh, level)
			ofh.write("( {} (span {} {}) (rel2par {})\n".format(nuc, beg, end, rel))
		l = node._childs[0]
		r = node._childs[1]
		print_dis_file(ofh, l, level + 1)
		print_dis_file(ofh, r, level + 1) 
		print_spaces(ofh, level)
		ofh.write(")\n")

def print_spaces(ofh, level):
	n_spaces = 2 * level
	for i in range(n_spaces):
		ofh.write(" ")

# print serial tree files

def print_serial_files(base_path, trees, outdir):
	path = create_dir(base_path, outdir)

	for tree in trees:
		outfn = path
		outfn += SEP
		outfn += tree._fname
		with open(outfn, "w") as ofh:
			print_serial_file(ofh, tree._root)

def print_serial_file(ofh, node, doMap=True):
	if node._type != "Root":
		nuc = node._nuclearity
		if doMap == True:
			rel = map_to_cluster(node._relation)
		else:
			rel = node._relation
		beg = node._span[0]
		end = node._span[1]
		ofh.write("{} {} {} {}\n".format(beg, end, nuc[0], rel))

	if node._type != "leaf":
		l = node._childs[0]
		r = node._childs[1]
		print_serial_file(ofh, l, doMap)
		print_serial_file(ofh, r, doMap)

def print_trees_stats(trees):
	rel_freq = defaultdict(int)

	for tree in trees:
		gen_tree_stats(tree._root, rel_freq)

	total = 0
	for _, v in rel_freq.items():
		total += v

	for k, v in rel_freq.items():
		rel_freq[k] = v / total

	print("relations frquencies: {}", rel_freq)

def gen_tree_stats(node, rel_freq):
	if node._type != "Root":
		nuc = node._nuclearity
		rel = map_to_cluster(node._relation)
		rel_freq[rel] += 1

	if node._type != "leaf":
		l = node._childs[0]
		r = node._childs[1]
		gen_tree_stats(l, rel_freq)
		gen_tree_stats(r, rel_freq)

def sent_splitter(trees, base_path, infiles_dir):
	if print_sents:
		if not os.path.isdir(sents_dir)

	for tree in trees:
		fn = tree._fname
		fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out", ""]) 
			
def gen_sentences(trees, base_path, infiles_dir):
	if print_sents:
		if not os.path.isdir(sents_dir):
   			os.makedirs(sents_dir)

   		with open(fn) as fh:
			# read the text
			inputstring = fh.read()
			all_sent = tokenize.sent_tokenize(inputstring)
			for sent in all_sent:
				if sent.strip() != '':
					
	for tree in trees:
		fn = tree._fname
		fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out", ""]) 
		with open(fn) as fh:
			# read the text
			content = ''
			lines = fh.readlines()
			for line in lines:
				if line.strip() != '':
					content += line
			sents = tokenize.sent_tokenize(content)
			for sent in sents:
				sent = sent.replace('\n', ' ')
				sent = sent.replace('  ', ' ')
				if sent.strip() == "\.":
					continue
				tree._sents.append(sent)

		if print_sents:
			fn_sents = build_file_name(tree._fname, base_path, sents_dir, "out.sents")
			with open(fn_sents, "w") as ofh:
				for sent in tree._sents[1:]:
					fh.write("{}\n".format(sent))

def build_infile_name(fname, base_path, dis_files_dir, suffs):
	for suf in suffs:
		fn = build_file_name(fname, base_path, dis_files_dir, suf)
		if os.path.exists(fn):
			return fn
	assert False, "File input does not exist: " +  \
		SEP.join([base_path, dis_files_dir, fname]) + \
		" with possible suffices " + "|".join(suffs)
	return None

def build_file_name(base_fn, base_path, files_dir, suf):
	fn = base_path
	fn += SEP
	fn += files_dir
	fn += SEP
	fn += base_fn
	if suf != '':
		fn += "."
		fn += suf
	return fn

def create_dir(base_path, outdir):
	remove_dir(base_path, outdir)
	path = base_path
	path += SEP
	path += outdir
	os.makedirs(path)
	return path

def remove_dir(base_path, dir):
	path = base_path
	path += SEP
	path += dir
	if os.path.isdir(dir):
		path_to_files = path
		path_to_files += SEP + "*"
		for fn in glob.glob(path_to_files):
			os.remove(fn)
		os.rmdir(path)

def check_path_separator(fn, last_dir_name):
	split_fn = fn.split(SEP)
	if not last_dir_name in split_fn:
		assert False, "Bad path separator was set: \"" + SEP + \
		"\" Call set_path_sep to change separator"

def set_path_sep(path_sep):
	global SEP
	SEP = path_sep

def set_print_stat(flag=True):
	global PRINT_STAT
	PRINT_STAT = flag