This project implemented various types of neural network (MLP, CNN, RNN) with clinical data for Epileptic Seizure Classification and Mortality Prediction. 

### Python and dependencies
In this project, we will work with PyTorch 1.3 on Python 3.6 environment. Please see environment.yml which contains a list of libraries needed to set environment for this project.
The given environment file is based on Linux (Ubuntu) system. Mac OS users who want to use GPU should build PyTorch from its source. Please refer to PyTorch installation guide. Windows users may need to install additional dependencies. Please refer to PyTorch official guide.

### Tasks
1. Epileptic Seizure Classification
- load_seizure_dataset in mydatasets.py loads the raw dataset files, and converts them into a PyTorch TensorDataset which contains data tensor and target (label) tensor.
- mymodels.py implemented a 3-layer MLP, a CNN model constituted by a couple of convolutional layers, pooling layer, and fully-connected layers and a Recurrent Neural Network (RNN).
- plots.py plots loss curves and accuracy curves for training and validation sets and a normalized confusion matrix for test set.
- train_seizure.py train and validate the model.

2. Mortality Prediction with RNN
In many realworld problems, however, data often contains variable-length of sequences, natural language processing for example. Also in healthcare problems, especially for longitudinal health records, each patient has a different length of clinical records history. In this problem, we will apply a recurrent neural network on variable-length of sequences from longitudinal electronic health record (EHR) data.

- etl_mortality_data.py implemented a pipeline that process the raw dataset to transform it to a structure that can be used with RNN model.
- mydatasets.py created a custom (inherited) PyTorch Dataset.
- MyVariableRNN class in mymodels.py implemented a RNN type that supports variable-length of sequences.
- train_variable_rnn.py train and validate the model and generated predictions to the Kaggle competition (https://www.kaggle.com/competitions/gt-cse6250-fall-2023-hw4/leaderboard). 