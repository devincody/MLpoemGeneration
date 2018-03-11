# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Flatten, Lambda
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from chartab import CharacterTable



# load ascii text and covert to lowercase
filename = "data/shakespeare.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# # create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
ct = CharacterTable(chars)
# char_to_int = dict((c, i) for i, c in enumerate(chars))
# int_to_char = dict((i, c) for i, c in enumerate(chars))
# # summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 40
vocab_len = len(ct.char_indices)
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 20):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	# print("sequences in: ", seq_in)
	# print("sequences out: ", seq_out)
	dataX.append(ct.encode(seq_in, seq_length))
	dataY.append(ct.encode(seq_out,1))
n_patterns = len(dataX)

print( "Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
# y = np_utils.to_categorical(dataY)
print("dataX", dataX[0], len(dataX[0]))
print("dataX", dataX[0][0], len(dataX[0][0]))
print("dataX", dataX[0][0][0])
# X = numpy.reshape(dataX, (n_patterns, seq_length, n_vocab))
# normalize
# X = X / float(n_vocab)
# one hot encode the output variable

# define the LSTM model

temperature = 1

model = Sequential()
model.add(LSTM(150, input_shape=(1, seq_length, vocab_len), return_sequences=True))
model.add(Dropout(0.1))
# model.add(LSTM(150))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Lambda(lambda x: x /temperature))
model.add(Dense(vocab_len, activation='softmax'))
# load the network weights
#filename = "weights-improvement-47-1.2219-bigger.hdf5"
#model.load_weights(filename)

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(dataX, dataY, epochs=10, batch_size=500, callbacks=callbacks_list)
# pick a random seed
# start = numpy.random.randint(0, len(dataX)-1)
seq_in = "shall i compare thee to a summer's day?\n"
pattern = [char_to_int[char] for char in seq_in]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	# x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	# print("Predict: ", prediction)
	# print("pattern: ", pattern)
	index = numpy.argmax(prediction)
	index = numpy.random.choice(range(len(prediction[0])), p=prediction[0])
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")
print("X ", X[0])