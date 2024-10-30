import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		# original MLP: input_dim = 178, output_dim = 5, hidden_layer = 16
		# self.fc1 = nn.Linear(178, 16)
		# self.fc2 = nn.Linear(16, 5)
		# modified MLP
		self.fc1 = nn.Linear(178, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 5)
		self.dropout1 = nn.Dropout(p=0.25)
		self.dropout2 = nn.Dropout(p=0.25)


	def forward(self, x):
		# original MLP function
		# x = F.sigmoid(self.fc1(x))
		# x = self.fc2(x)
		# modified MLP 
		x = F.relu(self.dropout1(self.fc1(x)))
		x = F.relu(self.dropout2(self.fc2(x)))
		# x = F.relu(self.fc1(x))
		# x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x



class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		# original CNN
		# self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		# self.pool = nn.MaxPool1d(kernel_size=2)
		# self.conv2 = nn.Conv1d(6, 16, 5)
		# self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		# self.fc2 = nn.Linear(128, 5)
		# modified CNN to avoid overfitting
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool1 = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(6, 8, 5)
		self.pool2 = nn.MaxPool1d(kernel_size=2)
		self.fc1 = nn.Linear(in_features=8 * 41, out_features=64)
		self.fc2 = nn.Linear(64, 5)
		self.dropout1 = nn.Dropout(p=0.25)
		self.dropout2 = nn.Dropout(p=0.25)


	def forward(self, x):
		# original CNN
		# x = self.pool(F.relu(self.conv1(x)))
		# x = self.pool(F.relu(self.conv2(x)))
		# x = x.view(-1, 16 * 41)
		# x = F.relu(self.fc1(x))
		# x = self.fc2(x)
		# modified CNN
		x = self.pool1(F.relu(self.dropout1(self.conv1(x))))
		x = self.pool2(F.relu(self.dropout2(self.conv2(x))))
		x = x.view(-1, 8 * 41)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		# original RNN
		# self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
		# self.fc = nn.Linear(in_features=16, out_features=5)
		# modified RNN
		self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True, dropout=0.25)
		self.fc = nn.Linear(in_features=16, out_features=5)


	def forward(self, x):
		# original RNN
		# x, _ = self.rnn(x)
		# x = self.fc(x[:, -1, :])
		# modified RNN
		x, _ = self.rnn(x)
		x = torch.tanh(x[:, -1, :])
		output = self.fc(x)
		return output

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		# original variableRNN
		# self.fc1 = nn.Linear(in_features=dim_input, out_features=32)
		# self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
		# self.fc2 = nn.Linear(in_features=16, out_features=2)
		# modified VariableRNN
		self.fc1 = nn.Linear(in_features=dim_input, out_features=16)
		self.fc2 = nn.Linear(in_features=16, out_features=8)
		self.rnn = nn.GRU(input_size=8, hidden_size=4, batch_first=True, num_layers=1)
		self.fc3 = nn.Linear(in_features=4, out_features=2)
		self.dropout1 = nn.Dropout(p=0.5)
		self.dropout2 = nn.Dropout(p=0.5)

	def forward(self, input_tuple):
		# original variableRNN
		# seqs, lengths = input_tuple
		# max_length = max(lengths)
		# seqs = torch.tanh(self.fc1(seqs))
		# packed_input = pack_padded_sequence(seqs, lengths, batch_first=True)
		# rnn_output, _ = self.rnn(packed_input)
		# padded_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=max_length)
		# seqs = self.fc2(padded_output[:, -1, :])
		# modified VariableRNN
		seqs, lengths = input_tuple
		fc_embedding = torch.tanh(self.fc1(seqs))
		fc_embedding = torch.tanh(self.fc2(self.dropout1(fc_embedding)))
		packed_input = pack_padded_sequence(self.dropout2(fc_embedding), lengths, batch_first=True)
		packed_out, hidden = self.rnn(packed_input)
		output = torch.squeeze(torch.tanh(self.fc3(hidden)))
		return output

	def _init_hidden(self, batch_size):
		hidden = torch.zeros(1, batch_size, self.hidden_size)
		return Variable(hidden)

