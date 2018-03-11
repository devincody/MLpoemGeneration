from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from chartab import CharacterTable

chartab = CharacterTable()

data_dim = 16
timesteps = 8
num_classes = len(chartab.chars)
batch_size = 32

f = open("data/shakespeare.txt")
text = ''.join(f.readlines())

for i in range(0, len(lext)-,10):


# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.

model = Sequential()
model.add(LSTM(150, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, num_classes)))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))