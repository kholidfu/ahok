# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/

import nltk
import csv
from textblob.classifiers import NaiveBayesClassifier
from nltk.corpus.reader import TaggedCorpusReader

reader = TaggedCorpusReader('.', 'idn.tsv')

print dir(reader)

