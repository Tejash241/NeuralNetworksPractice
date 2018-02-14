import numpy as np
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ======================= System constants =============================
USE_CUDA = torch.cuda.is_available()

# ========================== Model constants ===============================
N_CLASSES = 10
N_EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
VALID_SIZE = 0.

# ======================= Loading the data =============================
# def get_data(filename):
# 	with open(filename, mode='rb') as file:
# 		data = pickle.load(file)

# 	raw_images = data[b'data']
# 	cls = np.array(data[b'labels'])
# 	images = raw_images.reshape([-1, 3, 32, 32])
# 	return images, cls

# def load_training_data():
#     images = np.zeros(shape=[50000, 3, 32, 32], dtype=float)
#     cls = np.zeros(shape=[50000], dtype=int)

#     begin = 0
#     for i in range(5):
#         images_batch, cls_batch = get_data(filename="./cifar-10-batches-py/cifar-10-batches-py/data_batch_" + str(i + 1))
#         num_images = len(images_batch)
#         end = begin + num_images
#         images[begin:end, :] = images_batch
#         cls[begin:end] = cls_batch
#         begin = end

#     images = preprocess(images, cls)
#     plt.imsave('olo.png', images[0].transpose([1, 2, 0]))
#     print images.shape
#     return (images, cls)
transform = transforms.Compose(
    [transforms.ToTensor()])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py/', train=True,
                                        download=True, transform=transform)

num_train = len(trainset)
indices = list(range(num_train))
split = int(VALID_SIZE * num_train)

np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, 
                batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)

validloader = torch.utils.data.DataLoader(trainset, 
                batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py/', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# =========================== Data handling ================================
def preprocess(X, y):
	processed_X = np.zeros_like(X)
	for class_id in range(N_CLASSES):
		ids = y[np.where(y)==class_id]
		class_X = X[ids].copy()
		class_X -= np.mean(class_X, axis=1)
		class_X -= np.mean(class_X, axis=2)
		# class_X /= np.std(class_X)
		processed_X[ids] = class_X

	return processed_X

def make_one_hot(y):
	one_hot = np.zeros((y.shape[0], N_CLASSES))
	one_hot[np.range(y.shape[0]), y] = 1
	return one_hot

def get_next_batch(X, y):
	batch_X = []
	batch_y = []
	for i in range(y.shape[0]):
		batch_X.append(X[i])
		batch_y.append(y[i])
		if (i+1)%batch_size==0:
			yield(np.array(batch_X), np.array(batch_y))
			batch_X = []
			batch_y = []

# =========================== The model =====================================
class TVD(nn.Module):
	def __init__(self):
		super(TVD, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv1_bn = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
		self.conv2_bn = nn.BatchNorm2d(64)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64*6*6, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 512)
		self.fc4 = nn.Linear(512, 10)

	def forward(self, X):
		out = self.conv1(X)
		out = self.pool1(out)
		out = self.conv1_bn(out)
		out = F.relu(out)

		out = self.conv2(out)
		out = self.pool1(out)
		out = self.conv2_bn(out)
		out = F.relu(out)

		out = out.view(-1, 64*6*6)
		out = self.fc1(out)
		out = F.relu(out)
		out = F.dropout(out, p=0.4, training=self.training)

		out = self.fc2(out)
		out = F.dropout(out, p=0.4, training=self.training)

		out = self.fc3(out)
		out = F.dropout(out, p=0.4, training=self.training)

		out = self.fc4(out)
		return out

def weights_init(m):
	for p in m.modules():
	    if isinstance(p, nn.Conv2d): # xavier initialization
	        n = p.kernel_size[0] * p.kernel_size[1] * p.out_channels
	        p.weight.data.normal_(0, np.sqrt(2. / n))
	    elif isinstance(p, nn.BatchNorm2d):
	        p.weight.data.normal_(1.0, 0.02)
	        p.bias.data.fill_(0)

# ============================= Training ===========================
model = TVD().cuda() if USE_CUDA else TVD()
# model.apply(weights_init)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

losses = []
accs = []
for epoch in range(N_EPOCHS):
	for iteratn, data in enumerate(trainloader):
		X, y = data
		plt.imsave('olo.png', X[0].numpy().transpose([1, 2, 0]))
		exit()
		if USE_CUDA:
			X, y = Variable(X.cuda()), Variable(y.cuda())
		else:
			X, y = Variable(X), Variable(y)

		optimizer.zero_grad() # clears the gradients of all optimized Variable
		outputs = model(X)
		loss = criterion(outputs, y)
		loss.backward()
		optimizer.step()
		if iteratn%200==0:
			print 'Epoch', epoch, 'Iteration', iteratn, 'Loss', loss.data[0]

correct = 0
total = 0
model.eval()
for data in testloader:
    X, y = data
    X = X.cuda()
    y = y.cuda()
    outputs = model(Variable(X))
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()

print 'Accuracy of the network on the 10000 test images:', correct*1.0/total