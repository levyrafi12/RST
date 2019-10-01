import re
import copy
import glob
from collections import defaultdict
import os

from general import *

from utils import map_to_cluster
from relations_inventory import build_parser_action_to_ind_mapping

from dependency_graph import gen_and_load_dependency_parser
from preprocess_util import is_last_edu_in_sent

DEFAULT_TOKEN = 'transcend'
DEFAULT_TOKEN2 = 'immanent'

# debugging 
print_sents = True
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

	def get_span(self):
		return self._span[0], self._span[1]

class TreeInfo(object):
	def __init__(self):
		self._fname = '' # file name
		self._root = ''
		self._EDUS_table = [DEFAULT_TOKEN]
		self._sents = ['']
		self._edu_to_sent_ind = [0]
		self._sent_to_first_edu_ind = [0]
		self._edu_tokenized_table = [[DEFAULT_TOKEN]]
		self._edu_pos_tags_table = [['']]
		self._EDU_head_set = [[]]
		self._sent_tokenized_table = [[DEFAULT_TOKEN]]
		self._sent_pos_tags_table = [['']]
		# inputs of the EDUS seq encoder.
		self._edu_represent_table = [0] # EDU representation table
		# outputs of the EDUS seq encoder
		self._encoded_edu_table = [0]
		self._EDUS_parse = [{}] # data from dependency parser
		self._sents_parse = [{}] # data from dependency parser
		self._edus_seg_in_sent = [(0,0)] # segment boundaries

def preprocess(path, dis_files_dir, gen_dep=False, ser_files_dir='', bin_files_dir=''):
	build_parser_action_to_ind_mapping()

	trees = binarize_files(path, dis_files_dir, bin_files_dir)

	if ser_files_dir != '':
		print_serial_files(path, trees, ser_files_dir)

	for tree in trees:
		fn = build_infile_name(tree._fname, path, dis_files_dir, ["out.edus", "edus"])
		with open(fn) as fh:
			for edu in fh:
				edu = edu.strip()
				# edu = convert_edu(edu)
				tree._EDUS_table.append(edu)

	gen_sentences(trees)
	gen_and_load_dependency_parser(path, "dep_parse", trees, gen_dep)

	return trees

def binarize_files(base_path, dis_files_dir, bin_files_dir):
	trees = []
	path = base_path
	path += SEP
	path += dis_files_dir

	assert os.path.isdir(path), \
		"Path to dataset does not exist: " + dis_files_dir
	
	path += SEP + "*.dis"
	for fn in glob.glob(path):
		tree = binarize_file(fn, bin_files_dir)
		trees.append(tree)
	return trees

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
			# print("degree > 2")
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

	rel_freq_list = [(k,v) for k, v in rel_freq.items()]

	rel_freq_list = sorted(rel_freq_list, key=lambda elem: elem[1])
	rel_freq_list = rel_freq_list[::-1]
	print("most frequent relations: {}".format(rel_freq_list[0:5]))

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


def gen_sentences(trees):
    for tree in trees:
        edus_in_sent = []
        for edu_ind in range(1, len(tree._EDUS_table)):
            edu = tree._EDUS_table[edu_ind]
            edus_in_sent.append(edu)
            if (is_last_edu_in_sent(tree, edu_ind)):
                sent = ' '.join(edus_in_sent)
                tree._sents.append(sent)
                edus_in_sent = []

def set_print_stat(flag=True):
	global PRINT_STAT
	PRINT_STAT = flag

def convert_edu(edu):
	pos = edu.find("? -")
	if pos >= 0:
		edu = edu[:pos] + edu[pos + 1:]
	pos = edu.find("Corp. ")
	if pos >= 0:
		edu = edu[:pos + 4] + edu[pos + 5:]
	pos = edu.find('?"')
	if pos >= 0:
		edu = edu[:pos] + edu[pos + 1:]
	if edu[0].isdigit() and edu[1] == '.' and edu[2] == ' ' and edu[3].isupper():
		edu = edu[:1] + edu[2:]
	return edu