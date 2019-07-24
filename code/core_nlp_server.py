import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from nltk.parse.corenlp import CoreNLPServer
import os

STANFORD = os.path.join("..", "stanford-corenlp-full-2018-10-05")

server = CoreNLPServer(
	os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
	os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"))

server.start()