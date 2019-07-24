import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nltk.parse.corenlp import CoreNLPDependencyParser

parser = CoreNLPDependencyParser()
parse = next(parser.raw_parse("I put the book in the box on the table."))
print(parse)