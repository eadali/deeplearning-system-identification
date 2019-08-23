from keras.models import Sequential
from keras.layers import Dense, LSTM
from models import pendulum


x_data = numpy.random.rand(1000,10000)
pendulum_model = pendulum()

for input_signal in x_data:
  for _input in input_signal:
    output


model = Sequential()
model.add(LSTM(20,input_shape=(lahead, 1), batch_size=batch_size, stateful=stateful))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
