import tensorflow as tf
import numpy as np


import cPickle as pickle
import gzip




n_epochs = 50

bsize = 200

latent_dim = 2

lr = 3e-4


def get_mnist(filename='mnist.pkl.gz'):
	f = gzip.open('mnist.pkl.gz','rb')
	train_set, valid_set, test_set = pickle.load(f)
	f.close()
	return train_set, valid_set, test_set

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape,c=0.0):
	initial = tf.constant(c, shape=shape)
	return tf.Variable(initial)







print('building model')

X = tf.placeholder(tf.float32, shape=[None,784])
Y = tf.placeholder(tf.float32, shape=[None,784])

# Begin encoder

encoder1_W = weight_variable(shape=[784,1000])
encoder1_b = bias_variable(shape=[1000,])
encoder2_W = weight_variable(shape=[1000,500])
encoder2_b = bias_variable(shape=[500,])
encoder3_W = weight_variable(shape=[500,250])
encoder3_b = bias_variable(shape=[250,])
encoder4_W = weight_variable(shape=[250,30])
encoder4_b = bias_variable(shape=[30,])


encoder1 = tf.nn.relu(tf.matmul(X,encoder1_W) + encoder1_b)
encoder2 = tf.nn.relu(tf.matmul(encoder1,encoder2_W) + encoder2_b)
encoder3 = tf.nn.relu(tf.matmul(encoder2,encoder3_W) + encoder3_b)
encoder4 = tf.nn.relu(tf.matmul(encoder3,encoder4_W) + encoder4_b)




# vae latent z defn

mu_W = weight_variable(shape=[30,latent_dim])
mu_b = bias_variable(shape=[latent_dim,])
sigma_W = weight_variable(shape=[30,latent_dim])
sigma_b = bias_variable(shape=[latent_dim,])

z_mu = (tf.matmul(encoder4,mu_W) + mu_b)
z_sigma = tf.nn.softplus(tf.matmul(encoder4,sigma_W) + sigma_b)

#Qz = tf.contrib.bayesflow.stochastic_tensor.StochasticTensor(distributions.Normal(mu=z_mu, sigma=z_sigma))

Qz = tf.contrib.distributions.Normal(mu=z_mu, sigma=z_sigma)
Pz = tf.contrib.distributions.Normal(mu=np.zeros([latent_dim,],dtype='float32'),
		sigma=np.ones([latent_dim,],dtype='float32'))

z_sample = Qz.sample()

z_random = tf.placeholder_with_default(tf.random_normal([1,latent_dim]),
				shape=[None,latent_dim], name='default_latent_z')




# build decoder

decoder0_W = weight_variable(shape=[latent_dim,30])
decoder0_b = bias_variable(shape=[30])
decoder1_W = weight_variable(shape=[30,250])
decoder1_b = bias_variable(shape=[250,])
decoder2_W = weight_variable(shape=[250,500])
decoder2_b = bias_variable(shape=[500,])
decoder3_W = weight_variable(shape=[500,1000])
decoder3_b = bias_variable(shape=[1000,])
decoder4_W = weight_variable(shape=[1000,784])
decoder4_b = bias_variable(shape=[784,])

decoder0 = tf.nn.relu(tf.matmul(z_sample,decoder0_W) + decoder0_b)
decoder1 = tf.nn.relu(tf.matmul(decoder0,decoder1_W) + decoder1_b)
decoder2 = tf.nn.relu(tf.matmul(decoder1,decoder2_W) + decoder2_b)
decoder3 = tf.nn.relu(tf.matmul(decoder2,decoder3_W) + decoder3_b)
decoder4 = tf.matmul(decoder3,decoder4_W) + decoder4_b
x_hat = tf.nn.sigmoid(decoder4)

decoder0_ = tf.nn.relu(tf.matmul(z_random,decoder0_W) + decoder0_b)
decoder1_ = tf.nn.relu(tf.matmul(decoder0_,decoder1_W) + decoder1_b)
decoder2_ = tf.nn.relu(tf.matmul(decoder1_,decoder2_W) + decoder2_b)
decoder3_ = tf.nn.relu(tf.matmul(decoder2_,decoder3_W) + decoder3_b)
decoder4_ = tf.matmul(decoder3_,decoder4_W) + decoder4_b
output = tf.nn.sigmoid(decoder4_)


klloss = -(1)*tf.reduce_sum(1 + tf.log(z_sigma**2) - z_mu**2 - z_sigma**2,1)
#kldiv = tf.reduce_sum(tf.contrib.distributions.kl(Qz, Pz), 1)
sigmaloss = tf.reduce_sum((tf.ones_like(z_sigma)-z_sigma)**4 )

offset = 1e-7
obs = tf.clip_by_value(x_hat, offset, 1 - offset)
logloss = -1*(tf.reduce_sum(Y*tf.log(obs) + (1-Y)*tf.log(1-obs)))
#cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=decoder4))
cost = tf.reduce_mean(logloss + klloss + sigmaloss)


global_step = tf.Variable(0,trainable=False)

learning_rate = tf.train.exponential_decay(lr, global_step,1000,0.8,staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)




print('loading data')

train_set, valid_set, _ = get_mnist()
train_x, _ = train_set
valid_x, _ = valid_set
valid_x = valid_x[0:100]

n_batches = len(train_x) / bsize



def save_img(name='vae_demo.png'):
	from scipy.misc import imsave
	nx = 30
	ny = 30

	xvals = np.linspace(-9,9,nx)
	yvals = np.linspace(-9,9,ny)

	img = np.empty((28*ny,28*nx))

	for xi, xv in enumerate(xvals):
		for yi, yv in enumerate(yvals):
			z = np.array([[xv,yv]],dtype='float32')
			x_giv_z = output.eval(feed_dict={z_random:z})
			img[(nx-xi-1)*28:(nx-xi)*28,yi*28:(yi+1)*28] = x_giv_z[0].reshape(28,28)
	imsave(name,img)




print('training model for %i epochs' % (n_epochs))
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(n_epochs):
	logcost = logloss.eval(feed_dict={X:valid_x,Y:valid_x})
	kcost = klloss.eval(feed_dict={X:valid_x,Y:valid_x})
	zmean = z_mu.eval(feed_dict={X:valid_x})
	zsigm = z_sigma.eval(feed_dict={X:valid_x})
	print('epoch %i || log= %f | kdiv= %f mu= %f %f  sigma= %f %f' % (epoch+1, np.mean(logcost), np.mean(kcost),np.mean(zmean),np.std(zmean),np.mean(zsigm),np.std(zsigm)))
	for batch in range(n_batches):
		tx = train_x[batch*bsize:(batch+1)*bsize]
		tx = (tx > 0.5).astype('float32')
		train_step.run(feed_dict={X:tx,Y:tx})
        if epoch%5 == 0:
            save_img('pics/vae_%i.png'%(epoch))






# make picture of generated images

save_img('vae_demo.png')





















