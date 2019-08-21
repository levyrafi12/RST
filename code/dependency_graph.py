from nltk.parse.corenlp import CoreNLPDependencyParser

from general import *

def head_set_from_dependency_parse(base_path, files_dir, trees):
    parser = CoreNLPDependencyParser()

    path = base_path
    path += SEP
    path += files_dir

    if not os.path.isdir(path):
        create_dir(base_path, files_dir)

    for tree in trees:
        # print(tree._fname)
        sent_ind = 1
        edu_in_sent = []

        for edu_ind in range(1, len(tree._EDUS_table)):
            if tree._edu_to_sent_ind[edu_ind] == sent_ind:
                edu = tree._EDUS_table[edu_ind]
                edu_in_sent.append(edu)
            else:
                create_edus_head_set(parser, tree, edu_in_sent)
                sent_ind += 1
                edu_in_sent = []
                edu_in_sent.append(edu)

        create_edus_head_set(parser, tree, edu_in_sent)
        outfn = build_file_name(tree._fname, base_path, files_dir, "hs")

        with open(outfn, "w") as ofh:
            for edu_ind in range(1, len(tree._EDUS_table)):
                ofh.write("{}\n".format(tree._EDU_head_set[edu_ind]))

def create_edus_head_set(parser, tree, edu_in_sent):
    """
        Find head word set for each EDU in sentence. This set includes words whose 
        parent in dependency graph is ROOT or is not within the EDU.

    """ 
    sent = ''
    for edu in edu_in_sent:
        sent += edu
        sent += ' '

    parse = next(parser.raw_parse(sent))

    i = 1
    for edu in edu_in_sent:
        parse_edu = next(parser.raw_parse(edu))
        n_tok_in_edu = num_correct_tokens_in_edu(parser, parse, parse_edu, i)
        tree._EDU_head_set.append([])
        head_set = []
        high = i + n_tok_in_edu - 1
        low = i
        for j in range(low, high + 1):
            elem = parse.nodes[i]
            if elem['head'] < low or elem['head'] > high:
                tree._EDU_head_set[-1].append(elem['word'])
            i += 1
        # print(tree._EDU_head_set[-1])

def num_correct_tokens_in_edu(parser, parse_sent, parse_edu, ind_in_sent):
    """
        Return number of tokens in edu which are aligned to the sentence tokens
        Since parser consider edu as a sentence, it adds (or duplicates) in some cases 
        a dot as a new token (when dot attached to last token "Inc.")
    """
    n_tok = len(parse_edu.nodes)
    for ind_in_edu in range(1, n_tok):
        node = parse_edu.nodes[ind_in_edu]
        if node['word'] != parse_sent.nodes[ind_in_sent]['word']:
            break
        else:
            ind_in_edu += 1
            ind_in_sent += 1
   
    return ind_in_edu - 1


