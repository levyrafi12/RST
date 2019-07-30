from nltk.parse.corenlp import CoreNLPDependencyParser

def init_dependency_parser():
    return CoreNLPDependencyParser()

def dependency_parse_sent(parser, sent):
    parse = next(parser.raw_parse(sent))

    num_tokens = len(parse)
    for ind elem in parse:
        ind = ke
    {0: {'address': 0,
                 'ctag': 'TOP',
                 'deps': defaultdict(<class 'list'>, {'ROOT': [2]}),
                 'feats': None,
                 'head': None,
                 'lemma': None,
                 'rel': None,
                 'tag': 'TOP',
                 'word': None},
             1: {'address': 1,
                 'ctag': 'PRP',
                 'deps': defaultdict(<class 'list'>, {}),
                 'feats': '_',
                 'head': 2,
                 'lemma': 'I',
                 'rel': 'nsubj',
                 'tag': 'PRP',
                 'word': 'I'},