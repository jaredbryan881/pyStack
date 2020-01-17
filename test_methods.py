import scipy.io
import argparse
import matplotlib.pyplot as plt
import numpy as np

import pyStack

def main():
	# parse command line arguments
	parser = argparse.ArgumentParser(description="Tests and validates each stacking method")
	parser.add_argument('-i', '--infile', required=True, help="File path to the input noise waveforms")
	parser.add_argument('-f', '--filetype', required=False, default='mat', help="Type of data file to guide waveform reading, selected from: 'mat', 'hdf5'")
	parser.add_argument('-m', '--method', required=True, help="Select a stacking method to test from: 'linear', 'selective', 'pws', 'acf', or 'all'")
	parser.add_argument('-t', '--ccthreshold', required=False, default=0.0, help="Threhshold for the correlation coefficient used in selective stacking")
	parser.add_argument('-e', '--eps', required=False, default=1e-6, help="Convergence criteria for robust stacking")
	parser.add_argument('-x', '--maxiter', required=False, default=100, help="Maximum number of iterations performed in robust stacking")
	parser.add_argument('-p', '--power', required=False, default=2.0, help="Power for phase weighted stack")
	args = parser.parse_args()

	# read file from disk
	if args.filetype == 'mat':
		f_input = scipy.io.loadmat(args.infile)
		data = f_input["egfraw"]
		data = np.delete(data, 6, axis=1)
	elif args.filetype == 'hdf5':
		hf = h5py.File(args.infile, 'r')
	else:
		print("Something has gone wrong with reading the data from disk.")

	# dipatch to test stacking methods
	if args.method == 'linear':
		ref = pyStack.linear(data)
	elif args.method == 'selective':
		ref = pyStack.linear(data) # initial linear stack to get initial reference
		sel_ref, ar, ccArr = pyStack.selective(ref, data, args.ccthreshold) # first selective stacking iteration
		sel_ref2, ar, ccArr = pyStack.selective(sel_ref, data, args.ccthreshold) # second selective stacking iteration
	elif args.method == 'robust':
		pyStack.robust(data, args.eps, args.maxiter)
	elif args.method == 'pws':
		pyStack.pws(data, args.power)
	elif args.method == 'acf':
		pyStack.acf(data)
	else:
		print("Something has gone wrong with choosing a stacking method.")	


if __name__ == "__main__":
	main()
