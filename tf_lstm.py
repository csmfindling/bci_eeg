import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
from sys import stdout
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset

def extract_axis_1(data, ind):
	"""
	Get specified elements along the first axis of tensor.
	:param data: Tensorflow tensor that will be subsetted.
	:param ind: Indices to take (one for each element along axis 0 of data).
	:return: Subsetted tensor.
	"""

	batch_range = tf.range(tf.shape(data)[0])
	indices = tf.stack([batch_range, ind], axis=1)
	res = tf.gather_nd(data, indices)

	return res

def apply_to_zeros(lst, seq_maxlen, dtype=np.float32):
    nb_trajectories = len(lst)
    _, nbfeat       = lst[0].shape
    result = np.zeros([nb_trajectories, seq_maxlen, nbfeat], dtype)
    for idx_traj in range(nb_trajectories):
        result[idx_traj][:lst[idx_traj].shape[0],:lst[idx_traj].shape[1]] = lst[idx_traj]
    return result

class lstm():
	def __init__(self):

		# tensorflow session
		self.sess  = tf.Session()

		# Parameters
		self.batch_size  = 100
		self.seq_max_len = 20 # Sequence max length
		self.n_hidden    = 8  # hidden layer num of features
		self.n_classes   = 2  # linear sequence or not
		self.size_input  = 10
		# placeholders
		self.X         = tf.placeholder(tf.float32,[None, self.seq_max_len, self.size_input], name="x")
		self.Y         = tf.placeholder(tf.float32,[None, 2], name="y")
		self.seqlen    = tf.placeholder(tf.int32, [None])
		self.X1        = tf.placeholder(tf.float32,[1, 1, self.size_input], name="x1")
		self.C         = tf.placeholder(tf.float32,[1, self.n_hidden], name="C")
		self.M         = tf.placeholder(tf.float32,[1, self.n_hidden], name="M")
		# variables
		with tf.variable_scope("sigmoid") as scope:
			self.W         =  tf.Variable(tf.random_normal([self.n_hidden, 2]), trainable=True) #tf.get_variable('W1', shape=(6, 2), initializer=tf.contrib.layers.xavier_initializer(), trainable=True) # tf.Variable(tf.zeros([36, 16]), trainable=True) # 
			self.b         =  tf.Variable(tf.zeros([2]), trainable=True)
		# lstm
		with tf.variable_scope("rnn") as scope:
			self.cell                        = tf.contrib.rnn.LSTMCell(num_units=self.n_hidden, state_is_tuple=True)
			self.outputs, self.last_states   = tf.nn.dynamic_rnn(cell=self.cell, dtype=tf.float32, sequence_length=self.seqlen, inputs=self.X)
			scope.reuse_variables()
			self.output1, self.last_state1 = tf.nn.dynamic_rnn(cell=self.cell, dtype=tf.float32, sequence_length=[1], inputs=self.X1, initial_state=tf.contrib.rnn.LSTMStateTuple(self.C,self.M))
		self.last_output                  = extract_axis_1(self.outputs, self.seqlen - 1)
		# pred
		self.pred1     = tf.squeeze(tf.nn.softmax(tf.add(tf.matmul(tf.squeeze(self.output1, squeeze_dims=[0]), self.W), self.b)))
		self.pred      = tf.squeeze(tf.nn.softmax(tf.add(tf.matmul(self.last_output, self.W), self.b)))
		self.loss      = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.pred), reduction_indices=1)) + .05 * tf.nn.l2_loss(self.W) + .1 * tf.nn.l2_loss(tf.trainable_variables()[2])
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
		# initialize
		self.sess.run(tf.global_variables_initializer())
		self.saver         = tf.train.Saver()
		self.learning_done = False

	def train(self, n_datap, subj_idx, training_type):

		path           = 'data/subj{1}/leftright_{0}.hdf5'.format(training_type, subj_idx)

		n0, n_datap, split                    = 0, n_datap, .8
		n_train                               = int(n_datap * split) 
		train_set                             = H5PYDataset(path, which_sets=('train',), subset=slice(n0, n0 + n_train))
		train_stream                          = DataStream.default_stream(train_set, iteration_scheme=SequentialScheme(train_set.num_examples, self.batch_size))
		train_epoch_it                        = train_stream.get_epoch_iterator() 

		test_set                              = H5PYDataset(path, which_sets=('train',), subset=slice(n0 + n_train, n0 + n_datap))
		features_test, labels_test            = test_set.get_data(test_set.open(),slice(0, n_datap - n_train))
		feat_test_lstm                        = [features_test[(i - min(j, self.seq_max_len)):i] for i in range(1, n_datap - n_train + 1) for j in range(1, min(i + 1, self.seq_max_len + 1))]
		xlen_lstm_test                        = np.asarray([feat_test_lstm[i].shape[0] for i in range(len(feat_test_lstm))])
		labels_test_ltsm                      = np.concatenate([labels_test[i-1][np.newaxis]  for i in range(1, n_datap - n_train + 1) for j in range(1,min(i + 1, self.seq_max_len + 1))], axis=0)
		feat_test_lstm                        = apply_to_zeros(feat_test_lstm, self.seq_max_len) # zero padding


		idx_test_seq_max_len = np.where(xlen_lstm_test==self.seq_max_len)[0]
		feat_test_lstm       = feat_test_lstm[idx_test_seq_max_len]
		xlen_lstm_test       = xlen_lstm_test[idx_test_seq_max_len]
		labels_test_ltsm     = labels_test_ltsm[idx_test_seq_max_len]

# [i  for i in range(1, n_datap - n_train+1) for j in range(1,min(i+1, self.seq_max_len + 1))]
# apply_to_zeros(np.asarray([features_test[(i - min(j, self.seq_max_len)):i] for i in range(1, n_datap - n_train) for j in range(1,min(i+1, self.seq_max_len + 1))]), self.seq_max_len)
#
# for i in range(N):
# 	for j in range(i):
# 		features_test[(i - max(j, self.seq_max_len)):i]

		print('starting training \n')
		count      = 0
		train_loss = 0
		nb_epoch   = 0
		train_acc  = 0
		best_acc   = -1
		nb_batch   = 0
		patience   = 0
		while nb_epoch < 150 and patience < 100:
			try:
				stdout.write("."); stdout.flush()
				features, labels = train_epoch_it.__next__()
				nb_batch             += 1
				nb_features           = features.shape[0]
				feat_lstm             = [features[(i - min(j, self.seq_max_len)):i] for i in range(1, nb_features + 1) for j in range(1, min(i + 1, self.seq_max_len + 1))] 
				# apply_to_zeros(np.asarray([features[i:(i + self.seq_max_len)] for i in range(min(self.batch_size,nb_features))]), self.seq_max_len)
				xlen_lstm             = np.asarray([feat_lstm[i].shape[0] for i in range(len(feat_lstm))])
				# xlen_lstm             = np.asarray([features[i:(i + self.seq_max_len)].shape[0] for i in range(min(self.batch_size,nb_features))])
				lab_lstm              = np.concatenate([labels[i-1][np.newaxis]  for i in range(1, nb_features + 1) for j in range(1,min(i + 1, self.seq_max_len + 1))], axis=0)
				feat_lstm             = apply_to_zeros(feat_lstm, self.seq_max_len)
				# lab_lstm              = np.concatenate([labels[i + xlen_lstm[i] - 1][np.newaxis] for i in range(min(self.batch_size,nb_features))], axis=0)
				_,c,p                 = self.sess.run([self.optimizer, self.loss, self.pred], feed_dict={self.X:feat_lstm, self.Y:lab_lstm, self.seqlen:xlen_lstm})
				train_loss            = (c / nb_batch + train_loss * (nb_batch - 1)/nb_batch)
				train_acc             = np.mean(np.argmax(p, 1) == np.argmax(lab_lstm, 1))/nb_batch + train_acc * (nb_batch - 1)/nb_batch
			except StopIteration:
				nb_epoch     += 1
				tc, tp        = self.sess.run([self.loss, self.pred], feed_dict={self.X:feat_test_lstm, self.Y:labels_test_ltsm, self.seqlen:xlen_lstm_test})
				pred_acc      = np.mean(np.argmax(tp, 1) == np.argmax(labels_test_ltsm, 1))
				print('\n')
				print('finished {0} epoch'.format(nb_epoch))
				print('number of ones prediction is {0}'.format(np.sum(np.argmax(tp, 1) == 1)))
				print('test loss is {0}'.format(tc))
				print('test accuracy is {0}'.format(pred_acc))
				print('train loss is {0}'.format(train_loss))
				print('train accuracy is {0}'.format(train_acc))
				print ('new best accuracy {0}'.format(pred_acc))
				print('patience is {0}'.format(patience))
				if pred_acc > best_acc:
					patience   = 0
					best_acc   = pred_acc
					save_path  = self.saver.save(self.sess, "data/subj{1}/lstm_leftright_{0}.ckpt".format(training_type, subj_idx))
				else:
					patience += 1 #* (best_acc > .6)
				train_epoch_it = train_stream.get_epoch_iterator() 
				train_loss     = 0
				train_acc      = 0
				nb_batch       = 0
		print('\ntraining finished with accuracy {0}'.format(best_acc))
		self.saver.restore(self.sess, "./data/subj{1}/lstm_leftright_{0}.ckpt".format(training_type, subj_idx))
		self.learning_done=True

		print('number of ones in test set : {0}'.format(np.sum(labels_test_ltsm[:,0]==1)/len(labels_test_ltsm[:,0])))

		return [np.argmax(tp, 1), np.argmax(labels_test_ltsm, 1)]
		
	# c_state=np.random.rand(1,8); m_state=np.random.rand(1,8); datapoint=np.random.rand(1,1,14)
	def evaluate(self, c_state, m_state, datapoint):
		return self.sess.run([self.pred1, self.last_state1], feed_dict={self.X1:datapoint, self.C:c_state, self.M:m_state})



