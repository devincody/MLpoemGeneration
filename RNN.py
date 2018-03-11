#borrowed heavily from Keras example code

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

batch_sz = 500
n_epoch = 60

filename = "data/shakespeare.txt"
raw_text = open(filename).read()
text = raw_text.lower()

print('Poems length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of seq_length characters
seq_length = 60
stride = 1
sentences = []
next_chars = []
for i in range(0, len(text) - seq_length, stride):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])


print('nb sequences:', len(sentences))



x = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


print('Building Model')
model = Sequential()
model.add(LSTM(200, input_shape=(seq_length, len(chars))))# ,return_sequences=True))
# model.add(Flatten())
# model.add(Dense(1000))
# model.add(LSTM(40))#, input_shape=(seq_length, len(chars))))
# model.add(Dropout(0.03))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    if epoch %10 ==0 or epoch == n_epoch-1:
        print()

        start_index = random.randint(0, len(text) - seq_length - 1)
        for diversity in [0.25, 0.75, 1.5]:
            print('\nTemperature: ', diversity)

            generated = ''
            sentence = "shall i compare thee to a summer's day?\nthou art not love to"#text[start_index: start_index + seq_length]
            generated += sentence
            print('Seed: ' + sentence)
            sys.stdout.write(generated)

            for i in range(300):
                x_pred = np.zeros((1, seq_length, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=batch_sz,
          epochs=n_epoch, callbacks=[print_callback])

on_epoch_end(999, "fds'")