from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from models import pendulum
from numpy import cumsum
from numpy.random import rand


x_data = cumsum(cumsum(rand(1000,10000)-0.5, axis=1), axis=1)
y_data = numpy.zeros(x_data.shape)
pendulum_model = pendulum()


for sample_index, input_signal in enumerate(x_data):
  pendulum_model.reset()
  
  for time_index, _input in enumerate(input_signal):
    y_data[sample_index, time_index] = pendulum_model.update(_input)

    
    
pyplot.subplot(2,1,1)
pyplot.plot(x_data[0,:], label='input_signal')
pyplot.plot(2,1,2)
pyplot.plot(y_data[0,:], label='output_signal')
pyplot.grid()
pyplot.show()





model = Sequential()
model.add(LSTM(8,input_shape=(lahead, 1), batch_size=batch_size, stateful=stateful))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


epochs = 64
for epoch_index in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in data_input.
    # Each of these series are offset by one step and can be
    # extracted with data_input[i::batch_size].
    model.fit(x_train, y_train,
              batch_size=1, epochs=1,
              verbose=1, validation_data=(x_val, y_val),
              shuffle=False)
    model.reset_states()

y_pred = model.predict(x_test, batch_size=1)
  
pyplot.subplot(2,1,1)
pyplot.plot(x_test[0,:])
pyplot.plot(2,1,2)
pyplot.plot(y_test[0,:])
pyplot.plot(y_pred[0,:])
pyplot.show()


