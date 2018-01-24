import glob
import numpy as np
import re
import h5py
from waveletTransform import wavelet_transform
from fuel.datasets.hdf5 import H5PYDataset

# labels = np.random.randint(3, size=30)
def makehdf5(motor_signals, labels, subj_idx, M, training_type, motor_canals=np.arange(3)):
	dt             = .004
	canals         = motor_canals[:-1]
	ref            = motor_canals[-1]
	inter_time     = 1
	wt             = wavelet_transform()
	signals        = motor_signals[canals] - motor_signals[ref]
	y_train     = []
	X_train     = []
	count       = 0

	for n_idx in np.arange(int(M/2), int(signals.T.shape[0] - M/2), inter_time):
		if (np.sum(signals[:,int(n_idx - M/2):int(n_idx + M/2)])!=0):
			X_train.append(wt.transform(signals[:,int(n_idx - M/2):int(n_idx + M/2)]))
			try:
				y_train.append(labels[int(n_idx*dt)])
			except IndexError:
				print(int(n_idx*dt))
				y_train.append(labels[-1])
			count          += 1

	n_feat            = len(X_train[0])
	n_chan            = 1
	y_train           = np.asarray(y_train)
	print('processed %s examples '%str(count))
	print('number of labels 1: %s'%str(np.sum(y_train==1)))
	print('number of labels 2: %s'%str(np.sum(y_train==2)))
	print('number of features: %s'%str(n_feat * n_chan))

	n_total           = np.sum(y_train==1) + np.sum(y_train==2) + np.sum(y_train==0)
	assert(count == n_total)
	n_total_rightleft = np.sum(y_train==1) + np.sum(y_train==2)
	output_path       = 'data/subj{1}/leftright_{0}.hdf5'.format(training_type, subj_idx)
	h5file            = h5py.File(output_path, mode='w')
	hdf_features      = h5file.create_dataset('features', (n_total_rightleft, n_chan * n_feat), dtype='float32')
	hdf_labels        = h5file.create_dataset('labels', (n_total_rightleft, 2), dtype='uint16')

	hdf_features.dims[0].label   = 'batch'
	hdf_features.dims[1].label   = 'features'
	hdf_labels.dims[0].label     = 'batch'
	hdf_labels.dims[1].label     = 'labels'

	verif = 0
	for j in range(count):
		if y_train[j]==1 or y_train[j]==2:
			hdf_features[verif]     = np.asarray(X_train[j], dtype=np.float32)
			hdf_labels[verif]       = np.array([(y_train[j]==1)*1, (y_train[j]==2)*1], dtype=np.float32)
			verif                  += 1
	assert(verif == n_total_rightleft)

	# Save hdf5 train and submit count_val = 1298944 # n_total = 64641 + 64295
	split_dict = {}
	sources = ['features', 'labels']
	for name, slice_ in zip(['train'], [(0, n_total_rightleft)]):
	    split_dict[name] = dict(zip(sources, [slice_] * len(sources)))

	h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

	h5file.flush()
	h5file.close()

	print('hdf5 created')

	return n_total_rightleft