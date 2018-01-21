import input_data
import tensorflow as tf
from tensorflow.python.client import timeline
import time

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

start = time.time() 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

end = time.time() 
secs = end - start 
msecs = secs * 1000 # millisecs 
print( 'elapsed time: %f ms' %msecs)

sess = tf.InteractiveSession()

with tf.name_scope('inputs'):
	x = tf.placeholder("float", shape=[None, 784], name='x_in')
	y_ = tf.placeholder("float", shape=[None, 10], name='y_in')

	
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
for i in range(1000):
	batch = mnist.train.next_batch(2000)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print ("step %d, training accuracy %g"%(i, train_accuracy))
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	with tf.name_scope('train'):
		sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, options=options, run_metadata=run_metadata)
		end = time.time() 
		#print("elapsed time: %f ms" %((end-start)*1000))
		#train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, options=options, run_metadata=run_metadata)

print("Time taken: %f" % (time.time() - startTime))

# Create the Timeline object, and write it to a json file
fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline_01.json', 'w') as f:
	f.write(chrome_trace)
	
print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess = tf.Session() # get session
# tf.train.SummaryWriter soon be deprecated, use following
writer = tf.summary.FileWriter("logs/", sess.graph)