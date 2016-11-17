# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/

import nltk
from textblob.classifiers import NaiveBayesClassifier
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize

reader = TaggedCorpusReader('.', 'idn.tsv')

txt1 = """Presiden meresmikan kereta api super cepat Jakarta Bandung."""
sent_tokenize(txt1)
print word_tokenize(sent_tokenize(txt1)[0])

