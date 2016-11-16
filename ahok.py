from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob


train = [
    ('Kami selalu menyiram sayuran setiap hari', 'pos'),
    ('Dedi telah pergi sekolah pagi ini dengan tepat waktu', 'pos'),
    ('Anton adalah seorang pelajar yang rajin', 'pos'),
    ('Ayahku bukan seorang koruptor', 'pos'),
    ('Saya sudah memiliki cara untuk membuatnya luluh', 'pos'),
    ('Letusan gunung merapi tahun ini lebih besar dari tahun lalu', 'pos'),
    ('Cantik sekali wajah perempuan itu', 'pos'),
    ('Wajah cewek itu tidak cantik', 'neg'),
    ('Mereka adalah orang tua saya', 'pos'),
    ('Pak guru tidak jadi melaksanakan ulangan hari ini', 'neg'),
    ('Sepertinya dia tidak mempunyai uang yang cukup', 'neg'),
    ('Ayahnya bukan seorang dokter, melainkan cuma perawat saja', 'neg'),
    ('Aku ingin ikut, tapi Ibu tidak memberi izin', 'neg'),
    ('Walaupun kita jauh, tapi aku tak pernah sekalipun berniat meninggalkanmu', 'neg'),
]
test = [
    ('Istriku cantik', 'pos'),
    ('Istriku tidak cantik', 'neg'),
]

cl = NaiveBayesClassifier(train)

# Classify some text
# print(cl.classify("Istriku cantik"))  # "pos"
# print(cl.classify("Istriku tidak cantik."))   # "neg"

# Classify a TextBlob
blob = TextBlob("Istriku cantik. Istriku tidak cantik.",
                classifier=cl)
print(blob)
print(blob.classify())

for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features(5)
