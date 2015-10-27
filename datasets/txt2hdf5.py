import sys
import os
import h5py
import numpy
from fuel.datasets.hdf5 import H5PYDataset
import ipdb

class Text(H5PYDataset):
	# class to read text files and organize the batchs to use tbptt.
    def __init__(self, filename, which_sets,
    				   batch_size = 100, seq_size = 100,
    				   num_reiterations = 2, **kwargs):

		hdf5_path = filename.replace('.txt', '.hdf5')

		if not os.path.exists(hdf5_path):
			with open(filename, 'r') as f:
				num_chars =sum([len(line) for line in f])

			row_size = num_chars/batch_size
			no_minibatches = row_size/seq_size
			no_restarts = no_minibatches/num_reiterations

			no_restarts_test = no_restarts/20
			no_restarts_valid = no_restarts/20
			no_restarts_train = no_restarts - no_restarts_test - no_restarts_valid

			row_size = no_restarts*num_reiterations*seq_size
			no_minibatches = row_size/seq_size

			all_chars = ([chr(ord('a') + i) for i in range(26)] +
			             [chr(ord('A') + i) for i in range(26)] +
			             [chr(ord('0') + i) for i in range(10)] +
			             [',', '.', '!', '?', '<UNK>', '\n'] +
			             ['"', '"', ':', ';', '.', '-'] +
			             ['(', ')', ' ', '<S>', '</S>'])
			code2char = dict(enumerate(all_chars))
			char2code = {v: k for k, v in code2char.items()}

			unk_char = '<UNK>'

			features = numpy.empty((batch_size, row_size))
			with open(filename, 'r') as f:
				leftovers = numpy.array([])

				for row in xrange(batch_size):
					data_row = numpy.array([])
					data_row = numpy.hstack([data_row, leftovers])

					while len(data_row) < row_size:
						new_line = [char2code.get(x, char2code[unk_char]) for x in next(f)]
						data_row = numpy.hstack([data_row, new_line])
					
					leftovers = data_row[row_size:]
					data_row = data_row[:row_size]
					features[row] = data_row

			features_np = features

			h5file = h5py.File(hdf5_path, mode='w')

			features = h5file.create_dataset('features',
					   (no_minibatches*batch_size, seq_size),
			           dtype = numpy.dtype('int8'))

			for minibatch in xrange(no_minibatches):
				print (minibatch, no_minibatches)
				idx1 = numpy.arange(seq_size*minibatch,seq_size*(minibatch + 1))
				idx2 = numpy.arange(batch_size*minibatch,batch_size*(minibatch + 1))
				features[idx2,:] = features_np[:,idx1]

			features.dims[0].label = 'batch'
			features.dims[1].label = 'time'

			end_train = no_minibatches*batch_size - 2*batch_size*no_restarts_test*num_reiterations
			end_valid = no_minibatches*batch_size - batch_size*no_restarts_test*num_reiterations

			split_dict = {
			    'train': {'features': (0, end_train)},
			    'valid': {'features': (end_train, end_valid)},
			    'test': {'features': (end_valid, no_minibatches*batch_size)}
			    }

			h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

			h5file.flush()
			h5file.close()

		self.data_path = hdf5_path

		super(Text, self).__init__(self.data_path, which_sets, **kwargs)
