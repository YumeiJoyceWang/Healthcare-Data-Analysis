import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

import os
import pickle
import pandas as pd

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

	df = pd.read_csv(path)
	x = df.loc[:, 'X1':'X178'].values
	y = df['y'].values-1
	if model_type == 'MLP':
		data = torch.tensor(x.astype(np.float32))
		target = torch.tensor(y)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		dataset = TensorDataset(torch.from_numpy(x.astype('float32')).unsqueeze(1), torch.from_numpy(y))
	elif model_type == 'RNN':
		dataset = TensorDataset(torch.from_numpy(x.astype('float32')).unsqueeze(2), torch.from_numpy(y))
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	seq_lengths = list(map(max, seqs))
	max_length = max(seq_lengths)
	num = max(max_length) + 1
	return num


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit batch_seq
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		# ref: https://github.com/ast0414/pytorch-retain/blob/master/retain.py
		self.seqs = []
		for sequence in seqs:
			row = []
			col = []
			val = []
			for i, visit in enumerate(sequence):
				for code in visit:
					if code < num_features:
						row.append(i)
						col.append(code)
						val.append(1)
			self.seqs.append(sparse.coo_matrix((np.array(val), (np.array(row), np.array(col))), shape=(len(sequence), num_features)))

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

	# ref: https://github.com/ast0414/pytorch-retain/blob/master/retain.py	
	batch_seq, batch_label = zip(*batch)
	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda x: x.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_padded_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]
		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()
			
		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])
	
	seq_tensor = np.stack(sorted_padded_seqs, axis=0)
	seqs_tensor = torch.FloatTensor(seq_tensor)
	lengths_tensor = torch.LongTensor(sorted_lengths)
	labels_tensor = torch.LongTensor(sorted_labels)

	return (seqs_tensor, lengths_tensor), labels_tensor


if __name__ == '__main__':
	PATH_TRAIN_FILE = "data/seizure/seizure_train.csv"
	PATH_VALID_FILE = "data/seizure/seizure_validation.csv"
	PATH_TEST_FILE = "data/seizure/seizure_test.csv"

	# Path for saving model
	PATH_OUTPUT = "../output/seizure/"
	os.makedirs(PATH_OUTPUT, exist_ok=True)

	# Some parameters
	MODEL_TYPE = 'MLP'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
	NUM_EPOCHS = 200
	BATCH_SIZE = 64
	# USE_CUDA = False  # Set 'True' if you want to use GPU
	USE_CUDA = True  # Set 'True' if you want to use GPU
	NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

	device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
	torch.manual_seed(1)
	if device.type == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	train_dataset = load_seizure_dataset(PATH_TRAIN_FILE, MODEL_TYPE)


