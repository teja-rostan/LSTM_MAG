import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional


# create a cumulative sum sequence
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = np.random.random(size=n_timesteps)
    # calculate cut-off value to change class values
    limit = n_timesteps / 4
    # determine the class outcome for each item in cumulative sequence
    y = np.greater_equal(np.cumsum(X), limit)
    return X, y.astype(int)


# create multiple samples of cumulative sum sequences
def get_sequences(n_sequences, n_timesteps):
    seqX = np.zeros((n_sequences, n_timesteps))
    seqY = np.zeros((n_sequences, n_timesteps))
    # create and store sequences
    for i in range(n_sequences):
        X, y = get_sequence(n_timesteps)
        seqX[i] = X
        seqY[i] = y
    # reshape input and output for lstm
    seqX = seqX.reshape(n_sequences, n_timesteps, 1)
    seqY = seqY.reshape(n_sequences, n_timesteps, 1)
    return seqX, seqY


# define problem
n_timesteps = 10

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# train LSTM
X, y = get_sequences(50000, n_timesteps)
model.fit(X, y, epochs=1, batch_size=10)


# evaluate LSTM
X, y = get_sequences(100, n_timesteps)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100))


# make predictions
for _ in range(10):
    X, y = get_sequences(1, n_timesteps)
    yhat = model.predict_classes(X, verbose=0)
    exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)
    print('y=%s, yhat=%s, correct=%s' % (exp, pred, np.array_equal(exp, pred)))
