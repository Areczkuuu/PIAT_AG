import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Reshape
from keras.callbacks import Callback
from tensorflow.keras.layers import LSTM
import sys

tokenizacja = tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='-\n,...…!";?„', lower=True, split=' ', char_level=False, oov_token=None, analyzer=None,)

def generator_tekstu(model, dlugosc=100):
  poczatek_promptu = 'w dalekiej krainie za górami i lasami w zielonej dolinie'
  wyjscie, wzorzec = [tokenizacja.word_index[wartosc] for wartosc in poczatek_promptu.split()], [tokenizacja.word_index[wartosc] for wartosc in poczatek_promptu.split()]
  for i in range(dlugosc):
      x = np.reshape(wzorzec, (1, len(wzorzec), 1))
      indeks = np.argmax(model.predict(x, verbose=0))
      #wynik = tokenizacja.index_word[indeks]
      #sekwencja_wejsciowa = [tokenizacja.index_word[wartosc] for wartosc in wzorzec]
      wzorzec.append(indeks)
      wyjscie.append(indeks)
      wzorzec = wzorzec[1:len(wzorzec)]

  wyjscie_tekst = [tokenizacja.index_word[wartosc] for wartosc in wyjscie]

  czesc = " ".join(wyjscie_tekst)
  print(f"wygenerowany tekst:\n {czesc}\n")

# kod tworzony na podstawie poradnika: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
def usun_puste_listy(lista):
    return [x for x in lista if x]

class MojeCallback(Callback):
    def __init__(self, eks):
        super().__init__()
        self.eks = eks
    def on_epoch_end(self, epoka, logs=None):
        start = np.random.randint(0, len(self.eks) - 1)
        wzorzec = self.eks[start]
        for i in range(10):
            x = np.reshape(wzorzec, (1, len(wzorzec), 1))
            prognoza = model.predict(x, verbose=0)
            indeks = np.argmax(prognoza)
            #wynik = tokenizacja.index_word[indeks]
            #sekwencja_wejsciowa = [tokenizacja.index_word[wartosc] for wartosc in wzorzec]
            wzorzec.append(indeks)
            wzorzec = wzorzec[1:len(wzorzec)]

def przetwarzanie_danych(plik, dlugosc_sekwencji=10):
  tekst_surowy = open(plik, 'r', encoding='utf-8').read()
  tekst_surowy = tekst_surowy.split()

  tokenizacja.fit_on_texts(tekst_surowy)
  znaki = tokenizacja.word_counts
  sekwencja = tokenizacja.texts_to_sequences(tekst_surowy)
  #znak_do_int = tokenizacja.word_index
  n_vocab = len(znaki)

  #dlugosc_sekwencji = 10
  dataX = []
  dataY = []
  sekwencja = usun_puste_listy(sekwencja)
  n_znakow = len(sekwencja)

  for i in range(0, n_znakow - dlugosc_sekwencji, 1):
      sekwencja_wejsciowa = sekwencja[i:i + dlugosc_sekwencji]
      sekwencja_wyjsciowa = sekwencja[i + dlugosc_sekwencji]
      dataX.append([znak[0] for znak in sekwencja_wejsciowa])
      dataY.append(sekwencja_wyjsciowa[0])

  n_sekwencji = len(dataX)
  X = np.reshape(dataX, (n_sekwencji, dlugosc_sekwencji, 1))
  y = np.array(dataY)
  return X, dataX, y, n_vocab

def generator_danych(data, etykiety, rozmiar_partii):
    while True:
        indeksy = np.arange(len(data))
        np.random.shuffle(indeksy)
        data = data[indeksy]
        etykiety = etykiety[indeksy]

        for i in range(0, len(data), rozmiar_partii):
            partia_danych = data[i:i+rozmiar_partii]
            partia_etykiet = etykiety[i:i+rozmiar_partii]
            partia_etykiet = to_categorical(partia_etykiet, num_classes=39536)
            yield partia_danych, partia_etykiet


entry = przetwarzanie_danych(plik = "dataset.txt")
model = Sequential()
model.add(Embedding(input_dim=39536, output_dim=100, input_length=entry[0].shape[1]))
model.add(LSTM(128))
model.add(Dense(39536, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

wagi = "wagi_lstm.hdf5"
model.load_weights(wagi)
generator_tekstu(model, dlugosc=100)

# trening + zapisanie
# generator = generator_danych(entry[0].reshape(259108,10), entry[2], 32)
# model.fit(generator, steps_per_epoch=len(entry[0]) // 32, epochs=20, callbacks=[MojeCallback(eks = entry[1])])
# checkpoint = ModelCheckpoint("wagi_lst.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')