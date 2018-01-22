import input_data
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import numpy as np
import random

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
  
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_train(dir_name="MNIST_data/mnist_train.csv"):	
	
	print("Start reading", dir_name, "...")
	
	f = open(dir_name,'r')
	data=[]
	for line in f.readlines():
		data.append(line.strip().split(','))
	f.close()
	
	data = np.array(data).astype(np.float32)
	
	example = data[:,1:28*28+1]
	label = data[:,0]
	
	example /= 255
	
	label_onehot=np.zeros((len(label),10))
	
	for i in range(len(label)):
		label_onehot[i][int(label[i])] = 1
	
	#print(label[0:5])
	#print(label_onehot[0:5])
	
	#label = label.reshape(label.shape[0],1)
	
	label = label.reshape((-1, 1))
	
	return example, label

#
def get_line_offset(dir_name="MNIST_data/mnist_train.csv"):

	print("Start reading", dir_name, "...")
	f = open(dir_name,'r')
	
	# Read in the file once and build a list of line offsets
	line_offset = []
	offset = 0
	for line in f:
		line_offset.append(offset)
		offset += len(line)
	
	#print('line offst', line_offset)
	
	return line_offset


def get_example(index, line_offset, dir_name="MNIST_data/mnist_train.csv"):	
	
	#print("Start reading", dir_name, "...")
	
	f = open(dir_name,'r')
	
	#f.seek(index*(28*28+1),0)
	f.seek(line_offset[index])
	
	line = f.readline()
	#print(line)
	data = []
	data.append(line.strip('\n').split(','))
	#print(data)
	f.close()
	
	data = np.array(data).astype(np.float32)
	
	#print('data shape', data.shape)
	
	example = data[:,1:28*28+1]
	label = data[:,0]
	
	example /= 255
	
	label_onehot=np.zeros((len(label),10))
	
	for i in range(len(label)):
		label_onehot[i][int(label[i])] = 1
	
	return example[0], label_onehot[0]	

#Using get example
def get_batch2(line_offset, batch_size=200):

	example_batch=[]
	label_batch=[]

	
	for i in range(batch_size):
		index = random.randrange(0, 60000)
		#print(index)
		example, label_onehot = get_example(index, line_offset, dir_name="MNIST_data/mnist_train.csv")
		example_batch.append(example)
		label_batch.append(label_onehot)
		
	example_batch = np.array(example_batch).astype(np.float32)
	label_batch = np.array(label_batch).astype(np.float32)
	'''
	for i in range(batch_index*batch_size, (batch_index+1)*batch_size):
		example_batch.append(example[i])
		label_batch.append(label[i])
	'''	
	return example_batch, label_batch	


def get_batch(example, label, batch_size=200):

	example_batch=[]
	label_batch=[]

	
	for i in range(batch_size):
		index = random.randrange(0, len(example))
		#print(index)
		example_batch.append(example[index])
		label_batch.append(label[index])
		
	example_batch = np.array(example_batch).astype(np.float32)
	label_batch = np.array(label_batch).astype(np.float32)
	'''
	for i in range(batch_index*batch_size, (batch_index+1)*batch_size):
		example_batch.append(example[i])
		label_batch.append(label[i])
	'''	
	return example_batch, label_batch		

#start = time.time() 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#end = time.time() 
#secs = end - start 
#msecs = secs * 1000 # millisecs 
#print( 'elapsed time: %f ms' %msecs)

example, label = get_train()
line_offset = get_line_offset()
t_example, t_label = get_train("MNIST_data/mnist_test.csv")
sess = tf.InteractiveSession()

with tf.name_scope('inputs'):
	x = tf.placeholder("float32", shape=[None, 784], name='x_in')
	y_raw = tf.placeholder("uint8", shape=[None, 1])
	y_onehot = tf.cast(tf.one_hot(y_raw, depth=10),tf.float32)
	y_ = tf.reshape(y_onehot, (-1, 10))
	#y_ = tf.placeholder("float32", shape=[None, 10])
	
sess.run(tf.initialize_all_variables())

with tf.name_scope('conv_1'):
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,28,28,1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	
with tf.name_scope('pooling_1'):
	h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv_2'):
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	
with tf.name_scope('pooling_2'):
	h_pool2 = max_pool_2x2(h_conv2)
	
with tf.name_scope('flatten'):
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

with tf.name_scope('fully_1'):
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('drop_1'):
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('softmax'):
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Accuarcy'):
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

startTime = time.time() 
loadtime = 0
for i in range(1000):
	start_getbatch = time.time() 
	#batch = mnist.train.next_batch(200)
	batch = get_batch(example, label, 200)
	#batch = get_batch2(line_offset, 200)
	loadtime += (time.time()-start_getbatch)
	#print(batch[0], batch[1])
	if i%10 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_raw: batch[1], keep_prob: 1.0})
		test_accuracy = accuracy.eval(feed_dict={x: t_example, y_raw: t_label, keep_prob: 1.0})
		print ("step %d, training accuracy %g testing accuracy %g"%(i, train_accuracy, test_accuracy))
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	with tf.name_scope('train'):
		sess.run(train_step,feed_dict={x: batch[0], y_raw: batch[1], keep_prob: 0.5}, options=options, run_metadata=run_metadata)
		end = time.time() 
		#print("elapsed time: %f ms" %((end-start)*1000))
		#train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, options=options, run_metadata=run_metadata)

print("Total time: %f Loadtime: %f" % ((time.time() - startTime), loadtime))

# Create the Timeline object, and write it to a json file
fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline_01.json', 'w') as f:
	f.write(chrome_trace)
	
#print ("test accuracy %g"%accuracy.eval(feed_dict={x: example, y_: label, keep_prob: 1.0}))

sess = tf.Session() # get session
# tf.train.SummaryWriter soon be deprecated, use following
writer = tf.summary.FileWriter("logs/", sess.graph)