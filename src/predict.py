import numpy as np
import sys
import caffe
#predict_next_word('they', 'are', 'well', model, 10)
#myWordsInput = ['they','are','well'] #23 ,
#myWordsInput = ['are','well',','] #55 she
#myWordsInput = ['how','are','you'] #55 she
myWordsInput = ['they','had','to'] #55 she
#myWordsInput = ['there','are','times'] #55 she
#myWordsInput = ['you','like','him']
#myWordsInput = ['i','did','nt']
#myWordsInput = ['more','people','are']
myIndexInput = [-1,-1,-1]

vocabFile = open('csv/vocab.csv', "r")
myVocab = []
for line in vocabFile.readlines()[:]:
	word = line.split('=')[1].strip()
	myVocab.append(word)

# 3-gram words
myIndexInput[0] = myVocab.index(myWordsInput[0])+1
myIndexInput[1] = myVocab.index(myWordsInput[1])+1
myIndexInput[2] = myVocab.index(myWordsInput[2])+1

net = caffe.Net('model/deploy.prototxt','model_snapshot/snap_iter_3725000.caffemodel',caffe.TEST)
net.blobs['data'].data[...] = myIndexInput

print "Forward Prop with values %d %d %d - %s %s %s - %s %s %s" %(myIndexInput[0],myIndexInput[1],myIndexInput[2],
			myWordsInput[0],myWordsInput[1],myWordsInput[2],
			myVocab.index(myWordsInput[0]),myVocab.index(myWordsInput[1]),myVocab.index(myWordsInput[2]))

out = net.forward()

print "Word prediction: %s - %s" %(myVocab[out['prediction'].argmax()-1],out['prediction'].argmax()-1)
##top 5 predictions
top5Indexes = np.argsort(-out['prediction'][0])[:5].astype("int")
print top5Indexes
print "Top 5 predictions:"
for x in xrange(0, 5):
	print " %s" %(myVocab[top5Indexes[x]-1])
	print out['prediction'][0][top5Indexes[x]]

#print net.blobs
#print "EMBED"
#print net.params['ipWordEmbedding'][0].data
#print "HIDDEN"
#print net.params['ipHidden'][0].data
#print "RELU"
#print net.params['reluOutput'][0].data
#print "INPUT TO SOFTMAX"
#print net.params['inputToSoftmax'][0].data
#print "PREDICTION"
#print net.params['inputToSoftmax'][0].data