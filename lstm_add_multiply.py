import numpy as np
import tensorflow as tf

POWER = 32 # power 3 will produce xs [0, 3] for addition
BATCH_SIZE = 1024
LEARNING_RATE = 0.01
N_HIDDEN = 20
N_EPOCHS = 100
N_CLASSES = 2
N_ITER = 20
N_SUMMANDS = 2 # this means x1+x2, 3 would mean x1+x2+x3 (similar for multiplication)
tf.logging.set_verbosity(tf.logging.ERROR)

def get_one_hot(y):
	one_hot = np.zeros(y.shape+(2,))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			one_hot[i, j, int(y[i, j])] = 1

	return one_hot

def get_next_batch():
	X, y = np.zeros((POWER, BATCH_SIZE, N_SUMMANDS)), np.zeros((POWER, BATCH_SIZE))
	for i in range(BATCH_SIZE):
		num1 = np.random.randint(0, 2**(POWER-1)-1) # for multiplication, np.random.randint(0, 2**(POWER/2)-1)
		num2 = np.random.randint(0, 2**(POWER-1)-1) # for multiplication, np.random.randint(0, 2**(POWER/2)-1)
		X[:, i, 0] = map(int, list(format(num1, 'b').zfill(POWER)))[::-1]
		X[:, i, 1] = map(int, list(format(num2, 'b').zfill(POWER)))[::-1]
		y[:, i] = map(int, list(format(num1+num2, 'b').zfill(POWER)))[::-1]
	return X, get_one_hot(y)

def xavier_weights(shape, name):
	if len(shape)==2:
		N = shape[0]+shape[1]/2.0
	else:
		N = shape[0]

	return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=np.sqrt(1.0/N)))

with tf.Graph().as_default():
	cell = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN, state_is_tuple=True)
	init_state = cell.zero_state(BATCH_SIZE, tf.float32)
	X = tf.placeholder(tf.float32, (None, None, N_SUMMANDS))
	y = tf.placeholder(tf.float32, (None, None, N_CLASSES))
	wout = xavier_weights([N_HIDDEN, N_CLASSES], name='wout')
	bout = tf.Variable(tf.zeros([N_CLASSES]), name='bout')

	rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=True)

	last_layer = lambda lam: tf.add(tf.matmul(lam, wout), bout) # linear layer
	predictions = tf.map_fn(last_layer, rnn_outputs) # apply last_layer to each output of rnn_outputs
	# print predictions.get_shape(), 'predictions', y.get_shape()

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y, name='cross_entropy_loss'))
	correct_preds = tf.equal(tf.argmax(predictions, 2), tf.argmax(y, 2), name='correct_preds')
	accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(N_EPOCHS):
		    for _ in range(N_ITER):
				batch_X, batch_y = get_next_batch()
				preds, error, acc, _ = sess.run([predictions, loss, accuracy, optimizer], feed_dict={X: batch_X, y: batch_y})
				# print np.argmax(preds[:,0,:], axis=1) # this and below lines are purely for visualization
				# print np.argmax(batch_y[:,0,:], axis=1)
				# exit()
				print "Epoch {}, train error: {}, accuracy: {}".format(epoch, error, acc)

