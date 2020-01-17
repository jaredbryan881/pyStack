import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import time

def linear(data):
	"""Average array of CCFs along the station-pair axis.

	Args:
		:param data: np.array
			Cross-correlation waveforms, dimensions are [time, station-pair]

	Returns:
		:return stack: np.array
			Linearly stacked CCFs
	"""
	return np.mean(data, axis=1)
	
def selective(ref, data, ccthreshold):
	"""Filter stack by omitting low-similarity CCFs wrt a reference CCF.

	Args:
		:param ref: np.array
			Reference cross-correlation. Usually the long-term mean.
		:param data: np.array
			Cross-correlation waveforms, dimensions are [time, station-pair]
		:param ccthreshold: Float64
			Value of Pearson correlation coefficient below which to omit CCFs

	Returns:
		:return stack: np.array
			Linearly stacked CCFs after omission of low-similarity CCFs
	"""
	# compute the Pearson correlation coefficient for each CCF wrt. the reference
	ccArr = np.zeros(data.shape[1])
	stats = {}
	for i in range(ccArr.shape[0]):
		ccArr[i] = np.corrcoef(ref, data[:,i])[0,1]

	# get windows where cc > ccthreshold
	ind = np.where(ccArr>ccthreshold)
	ref_sel = np.mean(data[:, ind[0]], axis=1)

	# % of good traces
	acceptance_ratio =  (np.size(ind)/np.size(data, axis=1))*100

	return ref_sel, acceptance_ratio, ccArr

# vectorized correlation coefficient is slower than the loop...
def vcorrcoef(X,y):
	"""Vecorized correlation coefficient computation

	Args:
		:param X: np.array
			numpy ndarray containing windowed cross-correlations
		:param y: np.array
			1-d array containing reference cross-correlation

	Returns:
		:return r: np.array
			correlation coefficient for each windowed cross-correlation wrt the reference
	
	Reference: # from https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
	"""
	Xm = np.reshape(np.mean(X,axis=0),(1, X.shape[1]))
	ym = np.mean(y)
	y=np.reshape(y, (np.shape(y)[0], 1))

	r_num = np.sum((X-Xm)*(y-ym),axis=0)
	r_den = np.sqrt(np.sum((X-Xm)**2,axis=0)*np.sum((y-ym)**2))

	r = r_num/r_den
	return r

def robust(data, maxiter, eps):
	"""Iteratively perform a similarity-weighted stack following Pavlis and Vernon [2010].

	Args:
		:param data: np.array
			Cross-correlation waveforms, dimensions are [time, station-pair]
		:param maxiter: Int64
			Maximum number of iterations before outputting final stack
		:param eps: Float64
			Convergence criteria reached before outputting final stack

	Returns:
		:return stack: np.array
			Iteratively improved similarity-weighted stack

	Reference: Pavlis, G. L. and F. L. Vernon (2010). Array processing of teleseismic body waves with the USArray, Computers and Geosciences, 36(7) , pp. 910-920.

	"""
	N = np.size(data,axis=1)
	Bold = np.median(data, axis=1)
	Bold_norm = Bold / np.linalg.norm(Bold, ord=2)

	w = np.zeros(N)
	r = np.zeros(N)
	d2 = np.zeros(N)

	BdotD = np.zeros(N)
	# L2 norm for all columns in data
	for ii in range(N):
		d2[ii] = np.linalg.norm(data[:, ii], ord=2)
		# TODO: Vectorize dot product
		BdotD[ii] = np.dot(data[:, ii], Bold_norm)

		r[ii] = np.linalg.norm(data[:,ii] - (BdotD[ii] * Bold), ord=2)
		w[ii] = np.abs(BdotD[ii] / d2[ii] / r[ii])

	Bnew = np.average(data, axis=1, weights=w)
	Bnew_norm = Bnew / np.linalg.norm(Bnew, ord=2)

	# check convergence
	epsN = np.linalg.norm(Bnew_norm - Bold_norm, ord=1) / (np.linalg.norm(Bnew, ord=2)*N)
	Bold_norm = Bnew_norm
	count = 0

	# iteratively generate a weighted stack until eps is small or count is large
	while (epsN > eps) and (count <= maxiter):
		for ii in range(N):
			BdotD[ii] = np.dot(data[:,ii], Bold_norm)

			r[ii] = np.linalg.norm(data[:,ii] - (BdotD[ii] * Bold), ord=2)
			w[ii] = np.abs(BdotD[ii] / d2[ii] / r[ii])

		Bnew = np.average(data, axis=1, weights=w)
		Bnew_norm = Bnew / np.linalg.norm(Bnew, ord=2)

		# check convergence
		epsN = np.linalg.norm(Bnew_norm - Bold_norm, ord=1) / (np.linalg.norm(Bnew_norm, ord=2)*N)
		Bold_norm = Bnew_norm
		count += 1

	return Bnew

def pws(data, power):
	"""Perform a phase-weighted stack on array of time series

	Args:
		:param data: np.array
			Cross-correlation waveforms, dimensions are [time, station-pair]
		:param power: Float64
			Power for phase weighted stacking calcualtion

	Returns:
		:return phase weighted stack: np.array
			Phase weighted stack

	Reference: Martin Schimmel, Hanneke Paulssen, Noise reduction and detection of weak, coherent signals through phase-weighted stacks, Geophysical Journal International, Volume 130, Issue 2, August 1997, Pages 497–505, https://doi.org/10.1111/j.1365-246X.1997.tb05664.x
	"""
	Nrows,Ncols = np.shape(data)
	phase_stack = np.abs(np.mean(np.exp(1j * np.angle(data + scipy.signal.hilbert(data))), axis=1) / (Ncols**power))
	phase_stack /= np.max(np.abs(data), axis=1)

	return np.mean(np.dot(data, phase_stack[0]), axis=1)

def acf(data):
	print("Not yet implemented")