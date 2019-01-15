from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
import tensor as tf

x = tf.Variable([[1.,2.,3.],[3.,4.,5.]])
y = tf.Variable([[4.,5.,6.],[6.,7.,8.]])
z = Dot(axes=1)([x,y])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
sess.run(x)
sess.run(y)
sess.run(z)
