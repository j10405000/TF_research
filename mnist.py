import tensorflow as tf
import input_data
import time
import numpy as np


filename_queue = tf.train.string_input_producer(["MNIST_data/mnist_train.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0 for col in range(1)] for row in range(28*28+1)]
#print(record_defaults)
col = [[0 for col in range(1)] for row in range(28*28+1)]
col = tf.decode_csv(value, record_defaults=record_defaults)


x = tf.placeholder("float", shape=[None, 784])
y_raw = tf.placeholder("uint8", shape=[None, 1])
y_ = tf.one_hot(y_raw, depth=10, on_value=None, off_value=None, axis=None, dtype=None, name=None)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

W_fc1 = tf.Variable(tf.truncated_normal(shape=[784, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

dropout_rate = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)

W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y     = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

startTime = time.time()  
for i in range(100000):

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	#batch_xs, batch_ys = mnist.train.next_batch(1000)
	batch_xs, batch_ys = sess.run([[col[1:28*28+1]], [[col[0]]]])
	print(batch_xs, batch_ys)
	
	#print(mnist.train.next_batch(100)[1][1])
	train_step.run(feed_dict={x: batch_xs, y_raw: batch_ys, dropout_rate: 0.5})
	if(i % 100 == 0):
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_raw: batch_ys, dropout_rate: 1})
		#test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, dropout_rate: 1})
		#print ("step %d, training/testing accuracy: %g/%g"%(i, train_accuracy, test_accuracy))
		print ("step %d, training accuracy: %g"%(i, train_accuracy))
	
	coord.request_stop()
	coord.join(threads)
	
print("Time taken: %f" % (time.time() - startTime))
