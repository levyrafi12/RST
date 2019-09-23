import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nltk.parse.corenlp import CoreNLPDependencyParser

import pickle

def combine_parse_data(to_parse, from_parse):
	# print(to_parse)
	# print(from_parse)

	dist = len(to_parse) - 1
	print(dist)

	del from_parse[0]

	n_keys = len(from_parse)
	print(n_keys)

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

parser = CoreNLPDependencyParser()
text = "What about your track record? -- \" aren't asked of companies coming to market."

print(parse_text(parser, text))
 