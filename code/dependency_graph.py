from nltk.parse.corenlp import CoreNLPDependencyParser

from preprocess_util import is_last_edu_in_sent
from general import * 
import pickle 

def gen_and_load_dependency_parser(base_path, files_dir, trees, force_gen=False):
    path = concat_path(base_path, files_dir)

    if force_gen or not os.path.isdir(path):
        gen_dependency_parser(base_path, files_dir, trees)

    load_dependency_parser(base_path, files_dir, trees)

def gen_dependency_parser(base_path, files_dir, trees):
    """
        call dependency parser and dump parsing graphs to files
    """
    parser = CoreNLPDependencyParser()

    path = concat_path(base_path, files_dir)

    if not os.path.isdir(path):
        create_dir(base_path, files_dir)

    for tree in trees:
        edus_in_sent = []
        edus_parse = []
        sents_parse = []
        for edu_ind in range(1, len(tree._EDUS_table)):
            edu = tree._EDUS_table[edu_ind]
            edus_parse.append(parse_text(parser, edu))
            edus_in_sent.append(edu)
            if is_last_edu_in_sent(tree, edu_ind):
                sent = ' '.join(edus_in_sent)
                sents_parse.append(parse_text(parser, sent))
                # print("sent {} {}".format(sent, sents_parse[-1]))
                edus_in_sent = []

        outfn = build_file_name(tree._fname, base_path, files_dir, "dp.edus")

        with open(outfn, "wb") as handle:
            pickle.dump(edus_parse, handle)

        outfn = build_file_name(tree._fname, base_path, files_dir, "dp.sents")

        with open(outfn, "wb") as handle:
            pickle.dump(sents_parse, handle)

def load_dependency_parser(base_path, files_dir, trees):
    for tree in trees:
        # print("load dep parser {}".format(tree._fname))
        infn = build_infile_name(tree._fname, base_path, files_dir, ["dp.edus"])

        with open(infn, "rb") as handle:
            tree._EDUS_parse += pickle.load(handle)

        infn = build_infile_name(tree._fname, base_path, files_dir, ["dp.sents"])

        with open(infn, "rb") as handle:
            tree._sents_parse += pickle.load(handle)

        n_edus = len(tree._EDUS_table)
        first_edu_ind = 1 # index of first edu in sent
        sent_ind = 1

        for sent_parse in tree._sents_parse[1:]:
            sent_words = extract_parse_attr(sent_parse, 'word')
            tree._sent_tokenized_table.append(sent_words[1:])
            sent_lem_words = extract_parse_attr(sent_parse, 'lemma')
            tree._sent_lem_tokenized_table.append(sent_lem_words[1:])
            sent_pos_tags = extract_parse_attr(sent_parse, 'tag')
            tree._sent_pos_tags_table.append(sent_pos_tags[1:])
            assert first_edu_ind < n_edus, print("too few edus")

            for edu_ind in range(first_edu_ind, n_edus):
                tree._edu_to_sent_ind.append(sent_ind)
                edu = tree._EDUS_table[edu_ind]
                edu_parse = tree._EDUS_parse[edu_ind]
                is_new_sent = edu_ind == first_edu_ind
                l, h = set_edu_segment_in_sent(tree, sent_parse, edu_parse, is_new_sent)
                set_edu_head_set(tree, sent_ind, edu_ind)
                if is_last_edu_in_sent(tree, edu_ind):
                    assert tree._edus_seg_in_sent[-1][1] == len(sent_parse) - 1, \
                        print("bad partition of sent to edus: end pos {}, len sent {}". \
                            format(tree._edus_seg_in_sent[-1][1], len(sent_parse) - 1))
                    tree._sent_to_first_edu_ind.append(first_edu_ind)
                    first_edu_ind = edu_ind + 1
                    sent_ind += 1
                    break

def set_edu_head_set(tree, sent_ind, edu_ind):
    """
        Find the head word set for a given EDU in a sentence. This set includes words whose 
        parent in dependency graph is ROOT (a verb) or is not within the EDU.

    """ 
    sent_parse = tree._sents_parse[sent_ind]
    low, high = tree._edus_seg_in_sent[edu_ind]
    tree._EDU_head_set.append([])

    for i in range(low, high + 1):
        elem = sent_parse[i]
        if elem['head'] < low or elem['head'] > high or elem['rel'] == 'ROOT':
            tree._EDU_head_set[-1].append(elem['word'])

def set_edu_segment_in_sent(tree, sent_parse, edu_parse, is_new_sent):
    """
        Set edu segment boundaries in sentence
    """

    start_ind = 1 # scanning sent from position start ind

    if not is_new_sent:
        _, end_ind = tree._edus_seg_in_sent[-1] 
        start_ind = end_ind + 1

    ind_in_sent = start_ind
    n_tokens = len(edu_parse)
    edu_pos_tags = []
    edu_words = []
    edu_lem_words = []

    for ind_in_edu in range(1, n_tokens):
        sent_node = sent_parse[ind_in_sent]
        edu_node = edu_parse[ind_in_edu]

        # Since parser consider edu as a sentence, it adds (in fact duplicates) in some cases 
        # a dot as a new token (when dot attached to last token "Inc.")
        if sent_node['word'] == edu_node['word']:
            edu_words.append(edu_node['word'])
            edu_pos_tags.append(edu_node['tag'])
            edu_lem_words.append(edu_node['lemma'])
            ind_in_sent += 1

    tree._edus_seg_in_sent.append((start_ind, ind_in_sent - 1))
    tree._edu_tokenized_table.append(edu_words)
    tree._edu_lem_tokenized_table.append(edu_lem_words)
    tree._edu_pos_tags_table.append(edu_pos_tags)

    return start_ind, ind_in_sent - 1

def extract_parse_attr(parse, attr):
    attr_tuples = [(elem[attr], elem['address']) for _, elem in parse.items()]
    sorted_attr_tuples = sorted(attr_tuples, key=lambda elem: elem[1])
    return [elem[0] for elem in sorted_attr_tuples]

def combine_parse_data(to_parse, from_parse):
    dist = len(to_parse) - 1

    del from_parse[0]

    n_keys = len(from_parse)

    for key in range(1, n_keys + 1):
        to_parse[key + dist] = from_parse[key]
        del from_parse[key]
        elem = to_parse[key + dist]
        elem['head'] += dist

    return to_parse

def parse_text(parser, text):
    last_parse = None
    mult_parse = parser.parse_text(text)

    while True:
        try:
            parse = next(mult_parse)
            parse = dict(list(parse.nodes.items()))
            if last_parse == None:
                last_parse = parse
            else:
                last_parse = combine_parse_data(last_parse, parse)
        except StopIteration:
            break

    return last_parse

