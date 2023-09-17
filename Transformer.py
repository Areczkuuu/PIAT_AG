import numpy as np
import os
import string
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

liczba_wygenerowanych_tokenow = 40
poczatkowy_prompt = "w dalekiej krainie"
rozmiar_partii = 256
rozmiar_vocab = 39536
maxlen = 50
rozmiar_osadzenia = 256
liczba_glow = 4
rozmiar_warstwy_feedforward = 128


def maska_uczenia_przyczynowego(rozmiar_partii, n_cel, n_zrodlo, typ_danych):
    i = tf.range(n_cel)[:, None]
    j = tf.range(n_zrodlo)
    m = i >= j - n_zrodlo + n_cel
    maska = tf.cast(m, typ_danych)
    maska = tf.reshape(maska, [1, n_cel, n_zrodlo])
    mnoznik = tf.concat(
        [tf.expand_dims(rozmiar_partii, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(maska, mnoznik)

# kod tworzony na podstawie poradnika: https://keras.io/examples/generative/text_generation_with_miniature_gpt/
class BlokTransformatora(layers.Layer):
    def __init__(self, rozmiar_osadzenia, liczba_glow, rozmiar_ff, stopien=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(liczba_glow, rozmiar_osadzenia)
        self.ffn = keras.Sequential(
            [layers.Dense(rozmiar_ff, activation="relu"), layers.Dense(rozmiar_osadzenia),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(stopien)
        self.dropout2 = layers.Dropout(stopien)

    def call(self, wejscia):
        kształt_wejscia = tf.shape(wejscia)
        rozmiar_partii = kształt_wejscia[0]
        dlugosc_sekwencji = kształt_wejscia[1]
        maska_przyczynowa = maska_uczenia_przyczynowego(rozmiar_partii, dlugosc_sekwencji, dlugosc_sekwencji, tf.bool)
        wynik_atencji = self.att(wejscia, wejscia, attention_mask=maska_przyczynowa)
        wynik_atencji = self.dropout1(wynik_atencji)
        out1 = self.layernorm1(wejscia + wynik_atencji)
        wynik_ffn = self.ffn(out1)
        wynik_ffn = self.dropout2(wynik_ffn)
        return self.layernorm2(out1 + wynik_ffn)
val = random.randint(0,5)
class OsadzenieTokenowIPozycji(layers.Layer):
    def __init__(self, maxlen, rozmiar_vocab, rozmiar_osadzenia):
        super().__init__()
        self.token_osadzenie = layers.Embedding(input_dim=rozmiar_vocab, output_dim=rozmiar_osadzenia)
        self.pos_osadzenie = layers.Embedding(input_dim=maxlen, output_dim=rozmiar_osadzenia)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        pozycje = tf.range(start=0, limit=maxlen, delta=1)
        pozycje = self.pos_osadzenie(pozycje)
        x = self.token_osadzenie(x)
        return x + pozycje

def utworz_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    warstwa_osadzenia = OsadzenieTokenowIPozycji(maxlen, rozmiar_vocab, rozmiar_osadzenia)
    x = warstwa_osadzenia(inputs)
    blok_transformatora = BlokTransformatora(rozmiar_osadzenia, liczba_glow, rozmiar_warstwy_feedforward)
    x = blok_transformatora(x)
    outputs = layers.Dense(rozmiar_vocab)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    funkcja_straty = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[funkcja_straty, None],
    )
    return model

def niestandardowa_standardizacja(ciag_wejsciowy):
    male_litery = tf.strings.lower(ciag_wejsciowy)
    pozbawione_html = tf.strings.regex_replace(male_litery, "<br />", " ")
    return tf.strings.regex_replace(pozbawione_html, f"([{string.punctuation}])", r" \1")

def przygotuj_wejscia_i_etykiety_lm(text):
    text = tf.expand_dims(text, -1)
    tokenizowane_zdania = warstwa_wektoryzacji(text)
    x = tokenizowane_zdania[:, :-1]
    y = tokenizowane_zdania[:, 1:]
    return x, y

class GeneratorTekstu(keras.callbacks.Callback):
    def __init__(
        self, maksymalna_liczba_tokenow, tokeny_poczatkowe, indeks_do_slowa, top_k=10, co_ile_wypisywac=1
    ):
        self.maksymalna_liczba_tokenow = maksymalna_liczba_tokenow
        self.tokeny_poczatkowe = tokeny_poczatkowe
        self.indeks_do_slowa = indeks_do_slowa
        self.co_ile_wypisywac = co_ile_wypisywac
        self.k = top_k

    def wybierz_z(self, logity):
        logity, indeksy = tf.math.top_k(logity, k=self.k, sorted=True)
        indeksy = np.asarray(indeksy).astype("int32")
        przewidywane = keras.activations.softmax(tf.expand_dims(logity, 0))[0]
        przewidywane = np.asarray(przewidywane).astype("float32")
        return np.random.choice(indeksy, p=przewidywane)

    def detokenizuj(self, liczba):
        return self.indeks_do_slowa[liczba]

    def on_epoch_end(self, epoka, logi=None):
        tokeny_poczatkowe = [_ for _ in self.tokeny_poczatkowe]
        if (epoka + 1) % self.co_ile_wypisywac != 0:
            return
        liczba_wygenerowanych_tokenow = 0
        wygenerowane_tokeny = []
        while liczba_wygenerowanych_tokenow <= self.maksymalna_liczba_tokenow:
            pad_len = maxlen - len(tokeny_poczatkowe)
            indeks_probki = len(tokeny_poczatkowe) - 1
            if pad_len < 0:
                x = tokeny_poczatkowe[:maxlen]
                indeks_probki = maxlen - 1
            elif pad_len > 0:
                x = tokeny_poczatkowe + [0] * pad_len
            else:
                x = tokeny_poczatkowe
            x = np.array([x])
            y, _ = self.model.predict(x)
            wylosowany_token = self.wybierz_z(y[0][indeks_probki])
            wygenerowane_tokeny.append(wylosowany_token)
            tokeny_poczatkowe.append(wylosowany_token)
            liczba_wygenerowanych_tokenow = len(wygenerowane_tokeny)
        txt = " ".join(
            [self.detokenizuj(_) for _ in self.tokeny_poczatkowe + wygenerowane_tokeny]
        )
        print(f"Wygenerowany tekst:\n{txt}\n")

class GeneratorTekstu_przywczyt(keras.callbacks.Callback):
    def __init__(
        self, maksymalna_liczba_tokenow, tokeny_poczatkowe, indeks_do_slowa, model,top_k=10, co_ile_wypisywac=1
    ):
        self.maksymalna_liczba_tokenow = maksymalna_liczba_tokenow
        self.tokeny_poczatkowe = tokeny_poczatkowe
        self.indeks_do_slowa = indeks_do_slowa
        self.co_ile_wypisywac = co_ile_wypisywac
        self.k = top_k
        self.model = model

    def wybierz_z(self, logity):
        logity, indeksy = tf.math.top_k(logity, k=self.k, sorted=True)
        indeksy = np.asarray(indeksy).astype("int32")
        przewidywane = keras.activations.softmax(tf.expand_dims(logity, 0))[0]
        przewidywane = np.asarray(przewidywane).astype("float32")
        return np.random.choice(indeksy, p=przewidywane)

    def detokenizuj(self, liczba):
        return self.indeks_do_slowa[liczba]

    def on_epoch_end(self, epoka, logi=None):
        tokeny_poczatkowe = [_ for _ in self.tokeny_poczatkowe]
        liczba_wygenerowanych_tokenow = 0
        wygenerowane_tokeny = []
        while liczba_wygenerowanych_tokenow <= self.maksymalna_liczba_tokenow:
            pad_len = maxlen - len(tokeny_poczatkowe)
            indeks_probki = len(tokeny_poczatkowe) - 1
            if pad_len < 0:
                x = tokeny_poczatkowe[-maxlen:]
                indeks_probki = maxlen - 1
            elif pad_len > 0:
                x = tokeny_poczatkowe + [0] * pad_len
            else:
                x = tokeny_poczatkowe
            x = np.array([x])
            y, _ = self.model.predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                                      workers=1, use_multiprocessing=False, )
            wylosowany_token = self.wybierz_z(y[0][indeks_probki])
            while wylosowany_token == 0 :
                y, _ = self.model.predict(x[-(maxlen-val):], batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,)
                wylosowany_token = self.wybierz_z(y[0][indeks_probki])

            wygenerowane_tokeny.append(wylosowany_token)
            tokeny_poczatkowe.append(wylosowany_token)
            liczba_wygenerowanych_tokenow = len(wygenerowane_tokeny)
        txt = " ".join(
            [self.detokenizuj(_) for _ in self.tokeny_poczatkowe + wygenerowane_tokeny]
        )
        print(f"Wygenerowany tekst:\n{txt}\n")


nazwa_pliku = "./dataset.txt"
dane_tekstowe = tf.data.TextLineDataset(nazwa_pliku).shuffle(buffer_size=256).batch(rozmiar_partii)
warstwa_wektoryzacji = TextVectorization(standardize=niestandardowa_standardizacja, max_tokens=rozmiar_vocab - 1, output_mode="int", output_sequence_length=maxlen + 1,)
warstwa_wektoryzacji.adapt(dane_tekstowe)
slownik = warstwa_wektoryzacji.get_vocabulary()
dane_tekstowe = dane_tekstowe.map(przygotuj_wejscia_i_etykiety_lm).prefetch(tf.data.AUTOTUNE)

slowo_do_indeksu = {}
for indeks, slowo in enumerate(slownik):
    slowo_do_indeksu[slowo] = indeks

tokeny_poczatkowe = [slowo_do_indeksu.get(_, 1) for _ in poczatkowy_prompt.split()]

model = utworz_model()
zapisane_wagi = "wagi_transformatur.hdf5"
model.load_weights(zapisane_wagi)

start_tokens = [slowo_do_indeksu.get(_, 1) for _ in poczatkowy_prompt.split()]
Tgen_call = GeneratorTekstu_przywczyt(100, start_tokens, slownik,model)
Tgen_call.on_epoch_end(1,50)

# do nauki i zapisu modelu
# generator_tekstu_callback = GeneratorTekstu(liczba_wygenerowanych_tokenow, tokeny_poczatkowe, slownik)
# model.fit(dane_tekstowe, epochs=25, callbacks=[generator_tekstu_callback])
# sciezka="wagi_transformatur.hdf5"
# checkpoint = ModelCheckpoint(sciezka, monitor='loss', verbose=1, save_best_only=True, mode='min')
# model.fit(dane_tekstowe, verbose=2, epochs=2, callbacks=[checkpoint,generator_tekstu_callback])

