import tensorflow as tf 
import numpy as np 
import random 
import cPickle as pickle
import gzip




n_epochs = 10

bsize = 100




def get_mnist(filename='mnist.pkl.gz'):
	f = gzip.open('mnist.pkl.gz','rb')
	train_set, valid_set, test_set = pickle.load(f)
	f.close()
	return train_set, valid_set, test_set

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def sparse_weight_variable(shape, sparse=15):
	#initial = tf.truncated_normal(shape, stddev=0.1)
	indices = []
	values = []
	for i in range(sparse):
		indices.append([random.randint(0,shape[0]),random.randint(0,shape[1])])
		values.append(random.gauss(0,0.1))
	initial = tf.SparseTensor(indices=indices,values=values,dense_shape=shape)
	dense = tf.sparse_tensor_to_dense(initial,validate_indices=False)
	mask = tf.SparseTensor(indices=indices,values=[1]*sparse,dense_shape=shape)
	return tf.Variable(dense), mask

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)







print('building model')

X = tf.placeholder(tf.float32, shape=[None,784])
Y = tf.placeholder(tf.float32, shape=[None,784])

# Begin encoder

encode1_W, encode1_mask = sparse_weight_variable(shape=[784,1000],sparse=15)
encode1_b = bias_variable([1000])
encode2_W, encode1_mask = sparse_weight_variable(shape=[1000,500],sparse=15)
encode2_b = bias_variable([500])
encode3_W, encode1_mask = sparse_weight_variable(shape=[500,250],sparse=15)
encode3_b = bias_variable([250])


encode1 = tf.nn.relu(tf.matmul(X,encode1_W) + encode1_b)
encode2 = tf.nn.relu(tf.matmul(encode1,encode2_W) + encode2_b)
encode3 = tf.matmul(encode2,encode3_W) + encode3_b
encode3neuron = tf.nn.relu(encode3)

# Begin VAE z definition

mu_W = weight_variable(shape=[250,30])
mu_b = bias_variable([30])
logsd_W = weight_variable(shape=[250,30])
logsd_b = bias_variable([30])

mu = tf.matmul(encode3neuron,mu_W) + mu_b
logsd = tf.matmul(encode3,logsd_W) + logsd_b
sd = tf.exp(logsd)

var = tf.multiply(sd,sd)
meansq = tf.multiply(mu,mu)

kldiv = 0.5*meansq + 0.5*var - logsd - 0.5
klloss = tf.reduce_mean(kldiv)


noise = tf.random_normal(shape=tf.shape(sd), mean=0.0, stddev=1, dtype=tf.float32)
sdnoise = tf.multiply(sd, noise)

sample = sdnoise + mu


# Begin decoder 

decode4_W, decode4_mask = sparse_weight_variable(shape=[30,250],sparse=15)
decode4_b = bias_variable([250])
decode3_W, decode3_mask = sparse_weight_variable(shape=[250,500],sparse=15)
decode3_b = bias_variable([500])
decode2_W, encode1_mask = sparse_weight_variable(shape=[500,1000],sparse=15)
decode2_b = bias_variable([1000])
decode1_W, encode1_mask = sparse_weight_variable(shape=[1000,784],sparse=15)
decode1_b = bias_variable([784])


decode4 = tf.nn.relu(tf.matmul(sample,decode4_W) + decode4_b)
decode3 = tf.nn.relu(tf.matmul(decode4,decode3_W) + decode3_b)
decode2 = tf.nn.relu(tf.matmul(decode3,decode2_W) + decode2_b)
output = tf.matmul(decode2,decode1_W) + decode1_b

cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=output)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)




print('loading data')

train_set, valid_set, test_set = get_mnist()
train_x, _ = train_set
valid_x, _ = valid_set
valid_x = valid_x[0:100]

n_batches = len(train_x) / bsize




print('training model for %i epochs' % (n_epochs))
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(n_epochs):
	c = cost.eval(feed_dict={X:valid_x,Y:valid_x})
	print('epoch %i || valid cost = %f' % (epoch+1, np.mean(c)))
	for batch in range(n_batches):
		tx = train_x[batch*bsize:(batch+1)*bsize]
		train_step.run(feed_dict={X:tx,Y:tx})

























