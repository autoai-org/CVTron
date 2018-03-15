"""
Sample usage:
	python process_hico_labels.py 
"""

import argparse
import os

import scipy.io as sio


def generate_labels_file(anno_split, output_path):
	labels = []
	for data in (anno_split.T):
		indices = [i for i,v in enumerate(data) if v == 1]
		labels.append(indices)

	with open(output_path, 'w') as f:
		for row in labels:
			for item in row:
				f.write('{} '.format(item))
			f.write('\n')

def generate_filenames_file(filenames_split, output_path):
	with open(output_path, 'w') as f:
		for row in filenames_split:
			filename = map(str, row[0])
			filename = ''.join(filename)
			f.write('{}\n'.format(filename))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
			default='./hico_data', help='data directory')
	parser.add_argument('--anno_mat', type=str, 
			default='anno.mat', help='annotation mat file')
	args = parser.parse_args()
	
	data_dir = args.data_dir
	anno_file = os.path.join(data_dir, args.anno_mat)

	# Read the labels from annotation mat file
	anno_mat = sio.loadmat(anno_file)

	# Output labels
	anno_train = anno_mat['anno_train']
	labels_train_file = os.path.join(data_dir, 'labels_train.txt')
	generate_labels_file(anno_train, labels_train_file)
	anno_test = anno_mat['anno_test']
	labels_test_file = os.path.join(data_dir, 'labels_test.txt')
	generate_labels_file(anno_test, labels_test_file)

	# Output label_text
	list_action = anno_mat['list_action']
	label_text_file = os.path.join(data_dir, 'label_text.txt')
	with open(label_text_file, 'w') as f:
		for i, row in enumerate(list_action):
			obj = row['nname'][0]
			obj = ''.join(map(str, obj))
			verb = row['vname'][0]
			verb = ''.join(map(str, verb))
			f.write('{} {} {}\n'.format(i, obj, verb))

	# Output filenames
	filenames_train = anno_mat['list_train']
	filenames_train_file = os.path.join(data_dir, 'filenames_train.txt')
	generate_filenames_file(filenames_train, filenames_train_file)
	filenames_test = anno_mat['list_test']
	filenames_test_file = os.path.join(data_dir, 'filenames_test.txt')
	generate_filenames_file(filenames_test, filenames_test_file)
