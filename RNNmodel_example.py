import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


def create_RNN(hidden_unit, dense_unit, input_shape, activation):
    model=Sequential()
    model.add(SimpleRNN(hidden_unit, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_unit, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
demo_model = create_RNN(2,1,(3,1),activation=['linear','linear'])

wx = demo_model.get_weights()[0] # wx = Weights for the input(x) units
wh = demo_model.get_weights()[1] # wh = Weights for the hidden(h) units
bh = demo_model.get_weights()[2] # bh = Bias for the hidden units
wy = demo_model.get_weights()[3] # wy = Weight for the dense layer
by = demo_model.get_weights()[4] # by = Bias for the dense layer
 
print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)



x = np.array([1, 2, 3])
# Reshape the input to the required sample_size x time_steps x features 
x_input = np.reshape(x,(1, 3, 1))
y_pred_model = demo_model.predict(x_input)
 
 
m = 2
h0 = np.zeros(m)
h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1,wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2,wh) + bh
o3 = np.dot(h3, wy) + by
 
print('h1 = ', h1,'h2 = ', h2,'h3 = ', h3)
 
print("Prediction from network ", y_pred_model)
print("Prediction from our computation ", o3)