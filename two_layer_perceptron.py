import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ================= Global variables ========================
N_CLASSES = 10
N_HIDDEN = 64
INPUT_SIZE = 784
BATCH_SIZE = 128
LEARNING_RATE = 0.01
N_EPOCHS = 20
MOMENTUM_GAMMA = 0.9

# ================== Activation Functions ====================
def sigmoid(z):
	return 1./(1+np.exp(-z))

def sigmoid_prime(g_z):
	return g_z*(1-g_z)

def softmax(z): # z = (n,k)
	return (np.exp(z).T/np.sum(np.exp(z), axis=1)).T

# ===================== The Model ============================
def linear_model(W, b, X):
	return X.dot(W) + b

def numerical_first_layer_bw(X, W1, b1, W2, b2, y):
	eps = 0.01
	W1[0, 1] += eps
	dW1 = cross_entropy_with_logits(forward_pass(X, W1, b1, W2, b2)[1], y)
	W1[0, 1] -= 2*eps
	dW1 -= cross_entropy_with_logits(forward_pass(X, W1, b1, W2, b2)[1], y)
	dW1 /= 2*eps
	return dW1

# ==================== The gradients ===========================
def gradient_descent(u, du, n):
	return u+LEARNING_RATE*du/n

def sgd_momentum(u, du, du_minus1, n):
	if du_minus1 is None:
		return gradient_descent(u, du, n)
	else:
		du = MOMENTUM_GAMMA*du_minus1 + LEARNING_RATE*du
		return u + du/n

# ==================== The Training ============================
def forward_pass(X, W1, b1, W2, b2):
	z1 = linear_model(W1, b1, X)
	a1 = sigmoid(z1)
	z2 = linear_model(W2, b2, a1)
	a2 = softmax(z2)
	return a1, a2

def backward_pass(y, X, W1, b1, W2, b2, a1, a2):
	n_examples = y.shape[0]
	diff = y-a2
	dW2 = a1.T.dot(diff)
	db2 = np.sum(diff, axis=0)
	W2 = gradient_descent(W2, dW2, n_examples)
	b2 = gradient_descent(b2, db2, n_examples)

	common_part = diff.dot(W2.T)
	common_part = common_part * sigmoid_prime(a1)
	dW1 = X.T.dot(common_part)
	db1 = np.sum(common_part, axis=0)
	W1 = gradient_descent(W1, dW1, n_examples)
	b1 = gradient_descent(b1, db1, n_examples)
	return W1, b1, W2, b2

def backward_pass_momentum(y, X, W1, b1, W2, b2, a1, a2, dW1_minus1, db1_minus1, dW2_minus1, db2_minus1):
	n_examples = y.shape[0]
	diff = y-a2
	dW2 = a1.T.dot(diff)
	db2 = np.sum(diff, axis=0)
	W2 = sgd_momentum(W2, dW2, dW2_minus1, n_examples)
	b2 = sgd_momentum(b2, db2, db2_minus1, n_examples)

	common_part = diff.dot(W2.T)
	common_part = common_part * sigmoid_prime(a1)
	dW1 = X.T.dot(common_part)
	db1 = np.sum(common_part, axis=0)
	W1 = sgd_momentum(W1, dW1, dW1_minus1, n_examples)
	b1 = sgd_momentum(b1, db1, db1_minus1, n_examples)
	return W1, b1, W2, b2, dW1, db1, dW2, db2


# ===================== Evaluation metrics =====================
def get_accuracy(predictions, y):
	return np.sum(predictions==y)*1./y.shape[0]

def cross_entropy_with_logits(predictions, y): #y=predictions=(n, k)
	return -np.sum(y*np.log(predictions))*1./y.shape[0]

# ================== Data pre-processing =====================
def one_hot_vector(y):
	n_examples = y.shape[0]
	one_hot = np.zeros((n_examples, N_CLASSES))
	one_hot[range(n_examples), y] = 1
	return one_hot

def normalize_data(X):
	X = np.true_divide(X, 127.5)
	X -= 1
	return X

def get_mnist_data():
	mndata = MNIST('.')
	X, y = mndata.load_training()
	X_train, X_valid, y_train, y_valid = train_test_split(np.array(X), np.array(y), test_size=0.16666, random_state=42)
	X_test, y_test = mndata.load_testing()
	return X_train, y_train, X_valid, y_valid, np.array(X_test), np.array(y_test)

def get_batches(X, y):
	X_batch = []
	y_batch = []
	for i in xrange(y.shape[0]):
		X_batch.append(X[i])
		y_batch.append(y[i])
		i += 1
		if i%BATCH_SIZE==0:
			yield(np.array(X_batch), np.array(y_batch))
			X_batch = []
			y_batch = []


# ==================== Training part ============================
def train_model(X_train, y_train):
	# Step 1: pre-processing
	X_train = normalize_data(X_train)
	
	# Step 2: initialize weights and biases
	W1 = np.random.normal(loc=0.0, scale=1./np.sqrt(INPUT_SIZE), size=(INPUT_SIZE, N_HIDDEN))
	# W1 = np.random.randn(INPUT_SIZE, N_HIDDEN)*0.01
	b1 = np.zeros(N_HIDDEN)
	W2 = np.random.normal(loc=0.0, scale=1./np.sqrt(N_HIDDEN), size=(N_HIDDEN, N_CLASSES))
	# W2 = np.random.randn(N_HIDDEN, N_CLASSES)*0.01
	b2 = np.zeros(N_CLASSES)
	dW1 = None
	db1 = None
	dW2 = None
	db2 = None

	# Step 3: Iterate over all batches and epochs and monitor accuracy & loss
	loss_history = []
	accuracy_history = []
	for epoch in range(N_EPOCHS):
		X_train, y_train = shuffle(X_train, y_train)
		for iteratn, (batch_X, batch_y) in enumerate(get_batches(X_train, y_train)):
			# Step 4: Forward pass
			a1, a2 = forward_pass(batch_X, W1, b1, W2, b2)
			predictions = np.argmax(a2, axis=1)
			# Step 5: Evaluate
			loss = cross_entropy_with_logits(a2, one_hot_vector(batch_y))
			accuracy = get_accuracy(predictions, batch_y)
			loss_history.append(loss)
			accuracy_history.append(accuracy)
			if iteratn%100==0:
				print 'Epoch:', epoch, 'Iteration:', iteratn, 'Accuracy:', accuracy, 'Loss:', loss
				# print predictions, batch_y
			# Step 6: Backward pass
			W1, b1, W2, b2, dW1, db1, dW2, db2 = backward_pass_momentum(one_hot_vector(batch_y), batch_X, W1, b1, W2, b2, a1, a2, dW1, db1, dW2, db2)
			# W1, b1, W2, b2 = backward_pass(one_hot_vector(batch_y), batch_X, W1, b1, W2, b2, a1, a2)


	return W1, b1, W2, b2

# ======================= Helpers =================================================
X_train, y_train, X_valid, y_valid, X_test, y_test = get_mnist_data()
W1, b1, W2, b2 = train_model(X_train, y_train)
accuracies = []
X_test = normalize_data(X_test)
for iteratn, (batch_X, batch_y) in enumerate(get_batches(X_test, y_test)):
	a1, a2 = forward_pass(batch_X, W1, b1, W2, b2)
	predictions = np.argmax(a2, axis=1)
	loss = cross_entropy_with_logits(a2, one_hot_vector(batch_y))
	accuracy = get_accuracy(predictions, batch_y)
	# print predictions, batch_y
	print 'Testing accuracy', accuracy, 'loss', loss
	accuracies.append(accuracy)

print 'Final testing accuracy', np.mean(accuracies)
