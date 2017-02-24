import numpy as np
import sys
import caffe


vocabFile = open('csv/vocab.csv', "r")
myVocab = []
for line in vocabFile.readlines()[:]:
	word = line.split('=')[1].strip()
	myVocab.append(word)

vocab_size = 250
k = 10
myWordsInput = "do"

myIndexInput = myVocab.index(myWordsInput)




net = caffe.Net('model/deploy.prototxt','model_snapshot/snap_iter_3725000.caffemodel',caffe.TEST)
word_embedding_weights = net.params['ipWordEmbedding'][0].data
print word_embedding_weights.shape #250,50
word_rep = word_embedding_weights[myIndexInput,:]
print "Word Rep"
print word_rep
print word_rep.shape
a = np.array([[1,2,3],[4,5,6]])
m = np.asmatrix(a)
print a
print a.shape

print "SUM"
print a.sum(axis=1)

diff = word_embedding_weights - np.tile(word_rep, (vocab_size, 1));
print "WORD EMBEDING WEIGHTS"
print word_embedding_weights
print word_embedding_weights.shape
print "TILE"
print np.tile(word_rep, (vocab_size, 1))
print "DIFF"
print diff
print diff.shape
#innerPart = sum(np.power(diff, 2), 2)
powerPart = np.power(diff,2)
innerPart = powerPart.sum(axis=1)
distance = np.sqrt(innerPart);

#check indices
print distance
totalSize = len(distance)
print totalSize
top5Indexes = np.argsort(-distance)[:totalSize].astype("int")
print top5Indexes
print top5Indexes.shape

top5IndexesSmallest = top5Indexes[::-1] #reverse array

print top5IndexesSmallest

print "Closest words to: %s" %(myWordsInput)
print "Closest words:"
for x in xrange(0, k):
	print top5IndexesSmallest[x]
	print " %s" %(myVocab[top5IndexesSmallest[x]])
	print " %f" %(distance[top5IndexesSmallest[x]])


