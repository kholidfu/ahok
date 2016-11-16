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
    ('Wajah cewek itu tidak cantik.', 'neg'),
    ('Mereka adalah orang tua saya.', 'pos'),
    ('Pak guru tidak jadi melaksanakan ulangan hari ini.', 'neg'),
    ('Sepertinya dia tidak mempunyai uang yang cukup', 'neg'),
    ('Ayahnya bukan seorang dokter, melainkan cuma perawat saja', 'neg'),
    ('Aku ingin ikut, tapi Ibu tidak memberi izin', 'neg'),
    ('Walaupun kita jauh, tapi aku tak pernah sekalipun berniat '
     'meninggalkanmu', 'neg'),
    ('Dika memenangkan kejuaraan Bulu Tangkis tingkat dunia Di Inggris '
     'pada hari kamis lalu.', 'pos'),
    ('Ayah membeli sepeda motor baru seharga 50 juta rupiah di Dealer '
     'Yamaha beberapa hari yang lalu.', 'pos'),
    ('Para petani lada mengalami panen raya dan bahagia karena harga lada '
     'menyentuh 150 ribu per kilo gram.', 'pos'),
    ('Aisyah akan pergi ke Belanda bulan ini untuk belajar karena dia anak '
     'yang pintar sehingga mendapatkan beasiswa.', 'pos'),
    ('Koleksi hewan kebun binatang selamat mendapatkan koleksi tambahan '
     'karena Lia, si Induk Harimau Sumetra, melahirkan 3 ekor anak harimau '
     'kemarin malam.', 'pos'),
    ('Teluk Kiluan mendapatkan pengakuan sebagai tempat wisata paling indah '
     'untuk dikunjungi oleh majalah Our Trip Our Adventure.', 'pos'),
    ('Bulan depan Timnas Indonesia akan melaksanakan tour ke Negara-Negara '
     'Eropa dengan menjajal kemampuan beberapa klub sepak bola di sana.',
     'pos'),
    ('Program pemerintah untuk mementaskan kemiskinan di Indonesia mulai'
     'terlihat keberhasilannya berdasarkan data yang dikeluarkan oleh '
     'Kementerian Sosisal beberapa hari yang lalu.', 'pos'),
    ('Tahun depan Sumsang akan melemparkan produk keluaran terbarunya di '
     'pasar Indonesia dengan harga yang murah.', 'pos'),
    ('Mulai tahun depan Desa Suka Maju akan mendapatkan program listrik '
     'masuk desa dari pemerintah.', 'pos'),
    ('Pemerintah akan merealisasikan pembangunan jembatan selat sunda '
     'pada akhir tahun ini diawali dengan pembangunan-pembangunan '
     'tiang pancang awal di Provinsi Banten dan Lampung.', 'pos'),
    ('Sekolah SMAN 1 Ujung Tambak mendapatkan juara pertama lomba '
     'kebersihan tingkat kota dan akan mewakili provinsi dalam lomba '
     'sekolah bersih nasional tahun depan.', 'pos'),
    ('Shinta dan Rama akan membawa hubungan mereka ke tingkat '
     'selanjutnya setelah bersama-sama selama 10 tahun lamanya.', 'pos'),
    ('Aku akan diberikan hadiah sepeda baru oleh ayah karena '
    'menjadi juara kelas besok.', 'pos'),
    ('Nenek telah sembuh dari penyakitya dan telah kembali ke rumah '
     'setelah dirawat di rumah sakit selama kurang lebih dua minggu.',
     'pos'),
    ('Siti Zulaeha selamat dari kecelakaan tunggal yang menimpa '
     'dirinya di jalan raya pukul pada pukul 20. 00 wib kemarin malam.',
     'pos'),
    ('Malam ini Pak Raden mengundang kami ke acara selamatan '
     'beliau yang ingin pergi melaksanakan ibadah haji bulan yang '
     'akan datang.', 'pos'),
]
test = [
    ('Roni yang tidak memiliki agama akhirnya berangkat haji', 'pos'),
    ('Wajah cewek itu tidak cantik.', 'neg'),
]

cl = NaiveBayesClassifier(train)

# Classify some text
# print(cl.classify("Istriku cantik"))  # "pos"
# print(cl.classify("Istriku tidak cantik."))   # "neg"

# Classify a TextBlob
blob = TextBlob("Roni yang tidak memiliki agama akhirnya berangkat haji. "
                "Wajah cewek itu tidak cantik.",
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
