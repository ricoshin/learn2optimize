import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import utils

import pdb
_cuda_managers = {}


C = utils.getCudaManager('default')
C.set_cuda(torch.cuda.is_available())

class cnn_module(torch.nn.Module):
	def __init__(self):
		super(cnn_module, self).__init__()
		"""
		torch.nn.Conv2d(in_channels, out_channels,
				kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

		torch.nn.MaxPool2d(kernel_size, stride=None,
				padding=0, dilation=1, return_indices=False, ceil_mode=False)
		"""
		self.conv1 = nn.Conv2d(1, 20, 5, 1, 2) # 6@24*24
		# activation ReLU
		#self.pool1 = nn.MaxPool2d(3, 2) # 6@12*12
		self.conv2 = nn.Conv2d(20, 50, 5, 1, 2) # 16@8*8
		# activation ReLU
		#pool2 = nn.MaxPool2d(3, 2) # 16@4*4
		#self.fc1 = nn.Linear(4*4*50, 500)
		self.fc1 = nn.Linear(7*7*50, 500)
		#self.fc2 = nn.Linear(800, 500)
		self.fc3 = nn.Linear(500, 10)


	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		#pdb.set_trace()

		out = F.relu(self.fc1(out))
		#out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
batch_size = 128

train_set = dset.MNIST(root='/st1/jaehong/NIPS19/pytorch_tutorial', train=True, transform=trans, download=True)
test_set = dset.MNIST(root='/st1/jaehong/NIPS19/pytorch_tutorial', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
				 dataset=train_set,
				 batch_size=batch_size,
				 shuffle=True)
test_loader = torch.utils.data.DataLoader(
				dataset=test_set,
				batch_size=batch_size,
				shuffle=False)

x = torch.randn(batch_size, 1, 28, 28)
y = torch.randn(batch_size, 10)
print('==>>> total trainning batch number: %s'%format(len(train_loader)))
print('==>>> total testing batch number: %s'%format(len(test_loader)))


model = cnn_module().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


for epoch in range(10):
	# trainning
	ave_loss = 0
	for batch_idx, (x, target) in enumerate(train_loader):
		optimizer.zero_grad()
		x, target = x.cuda(), target.cuda()
		"""
		if use_cuda:
			x, target = x.cuda(), target.cuda()
		x, target = Variable(x), Variable(target)
		"""
		out = model(x)
		loss = criterion(out, target)
		ave_loss = ave_loss * 0.9 + loss.item() * 0.1
		loss.backward()
		optimizer.step()
		if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
			print('==>>> epoch: %s, batch index: %s, train loss: %.6f'%(
				epoch, batch_idx+1, ave_loss))
	# testing
	correct_cnt, ave_loss = 0, 0
	total_cnt = 0
	for batch_idx, (x, target) in enumerate(test_loader):
		x, target = x.cuda(), target.cuda()
		out = model(x)
		loss = criterion(out, target)
		_, pred_label = torch.max(out.data, 1)
		total_cnt += x.data.size()[0]

		correct_cnt += (pred_label == target.data).sum()
		# smooth average
		ave_loss = ave_loss * 0.9 + loss.item() * 0.1

		if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
			print('==>>> epoch: %s, batch index: %s, test loss: %.6f, acc: %.3f'%(
				epoch, batch_idx+1, ave_loss, float(correct_cnt) / total_cnt))
