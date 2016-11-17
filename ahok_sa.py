from textblob.classifiers import NaiveBayesClassifier

train = [
    ('Ikut gelar perkara Ahok, Neno Warisman: Dari bahasa terbukti ada penistaan.', 'neg'),
    ('Soal gelar perkara Ahok, Habib Rizieq: Sementara kita lihat baik.', 'neg'),
    ('Ikut blusukan di pasar, Cathy Sharon: Mending naik bajaj deh', 'neu'),
    ('Gelar perkara ahok, ICMI: Apapun hasilnya sikapi dengan damai dan lapang dada.', 'pos'),
    ('MUI: Klarifikasi ke Ahok tidak menjadi keharusan', 'neu'),
    ('Ahok: Saya sudah diminta mundur dari Pilgub DKI 2017', 'neg'),
    ('Bagaimana jika Ahok jadi tersangka? Ini kata KPU DKI', 'neg'),
    ('Misteri Ahok Diminta mundur dari Pilgub DKI', 'neu'),
    ('Jika tak ditemukan bukti, Ahok tak dapat dilaporkan kembali', 'pos'),
    ]
test = [
    ('Ahok kalah.', 'neg')
    ]
cl = NaiveBayesClassifier(train)

print cl.classify('Ahok kalah.')
print cl.accuracy(test)
print cl.show_informative_features(5)
