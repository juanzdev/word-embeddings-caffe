import h5py, os
import numpy as np
import csv


trainXFilePath = 'csv/train_x.csv'
trainTFilePath = 'csv/train_t.csv'
testXFilePath = 'csv/test_x.csv'
testTFilePath = 'csv/test_t.csv'

'''
trainXFile = 'smalltrain_x.csv'
trainTFile = 'smalltrain_t.csv'
testXFile = 'smalltest_x.csv'
testTFile = 'smalltest_t.csv'
'''

#read csv training data
with open(trainXFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

#shape (372550, 3)
data = np.array(data_as_list).astype("int")

#read csv training data labels
with open(trainTFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

#shape (372550, 1)
labels = np.array(data_as_list).astype("int")

#create HDF5 file with training data and labels
f = h5py.File('hdf5/train.h5', 'w')
f.create_dataset('data', data = data)
f.create_dataset('label',  data = labels)
f.close()

#read csv test data
with open(testXFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

data = np.array(data_as_list).astype("int")

#read csv test data labels
with open(testTFilePath,'rb') as f:
	reader = csv.reader(f)
	data_as_list = list(reader)

labels = np.array(data_as_list).astype("int")

#create HDF5 file with test data and labels
f = h5py.File('hdf5/test.h5', 'w')
f.create_dataset('data', data = data)
f.create_dataset('label',  data = labels)
f.close()
