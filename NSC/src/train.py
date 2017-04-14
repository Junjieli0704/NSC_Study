
#-*- coding: UTF-8 -*-  
import sys
from Dataset import *
from LSTMModel import LSTMModel




def get_init_file(dataname):
	file_fold = '../../../NSC-data/'
	word_list_file = file_fold + dataname + '/wordlist.txt'
	train_set_file = file_fold + dataname + '/train.txt'
	dev_set_file = file_fold + dataname + '/dev.txt'
	model_file = '../model/'+dataname+'/bestmodel'
	return word_list_file,train_set_file,dev_set_file,model_file


if __name__ == '__main__':

	#dataname = sys.argv[1]
	#classes = sys.argv[2]
	dataname = 'yelp13'
	classes = '5'
	word_list_file,train_set_file,dev_set_file,model_file = get_init_file(dataname)
	voc = Wordlist(word_list_file)
	trainset = Dataset(train_set_file, voc)
	devset = Dataset(dev_set_file, voc)
	print 'data loaded.'

	model = LSTMModel(voc.size,trainset, devset, dataname, classes, None)
	model.train(100)
	print '****************************************************************************'
	print 'test 1'
	result = model.test()
	print '****************************************************************************'
	print '\n'
	for i in xrange(1,400):
		model.train(100)
		print '****************************************************************************'
		print 'test',i+1
		newresult=model.test()
		print '****************************************************************************'
		print '\n'
		if newresult[0]>result[0] :
			result=newresult
			model.save(model_file)
	print 'bestmodel saved!'

