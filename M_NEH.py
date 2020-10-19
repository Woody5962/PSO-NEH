import numpy as np
import random
from Gatt_Figure import gatt


def generate_A(p, PT):
	"""
	generate A from solution p and processing time matrix PT
	:param p: solution p
	:param PT: processing time matrix
	:return: a processing time matrix of solution p
	"""
	rows, cols = np.shape(p)
	A = np.zeros([rows, cols])
	for job in range(rows):
		for procedure in range(cols):
			A[job, procedure] = PT[job, p[job, procedure]]
	return A


def processing_time(k, A):
	"""
	calculate the processing time of job k
	:param k: a job's No.
	:param A: the processing_time matrix
	:return: the processing_time of job_k
	"""
	m = np.shape(A)[1]
	pt = 0  # initialize the processing_time
	for j in range(m):
		if A[k, j] != 0:
			pt += A[k, j]
	return pt


def parallel(k, p):
	"""
	calculate a parallel job of job k or assert none
	:param k: a job's No.
	:param p: a solution from PSO
	:return: a No. of a parallel machine or None
	"""
	parallel_machine = []
	n, m = np.shape(p)
	for i in range(n):
		count = 0
		if i != k:
			for j in range(m):
				if p[i][j] != p[k][j]:
					count += 1
				else:
					break
			if count == m:
				parallel_machine.append(i)
	if len(parallel_machine) == 0:
		return None
	else:
		pm_no = random.randint(0, len(parallel_machine)-1)  # the no. of the parallel_machine
		return parallel_machine[pm_no]


def conflict_check(k, p, S, R):
	"""
	calculate the conflict time to delay in order to avoid the conflict
	:param k: a job's no.
	:param p: the matrix of machine schedule
	:param S: the matrix of start time
	:param R: the release time of every machine
	:return: the max conflict time
	"""
	m = np.shape(S)[1]
	D_value = [0]  # initialize the D_value,if no element is appended later, 0 is what we need
	for j in range(m):
		if R[p[k, j]] <= S[k, j]:
			continue
		else:
			D_value.append(R[p[k, j]] - S[k, j])
	D_value.sort(reverse=True)
	return D_value[0]


def complete_max(k, conflict_time, A, C, sequence):
	"""
	calculate the C_MAX if we choose job_k
	:param sequence: the jobs which are already done
	:param k: job k
	:param conflict_time: conflict_time （delay——time）
	:param A: the processing time matrix
	:param C: the completing time matrix
	:return: the C_max
	"""
	p_time = processing_time(k, A)  # processing time
	k_max = conflict_time + p_time  # job_k's Complete time
	C_max = k_max
	for i in sequence:
		if C[i, -1] > C_max:
			C_max = C[i, -1]
	return C_max


def m_neh(p, *args, **kwargs):
	"""
	calculate the obj_value and the sequence of job
	:param args: args passed from the partial function
	:param p: a partial solution
	:return: the obj_value
	"""
	# A: the processing time matrix
	# mop: amount of machines in each procedure
	# debug: print the sequence or not
	if len(args) == 3:
		PT, mop, display = args
		fg = -1
	elif len(args) == 4:
		PT, mop, display, fg = args
	else:
		print('Wrong parameters for m_neh!')
	A = generate_A(p, PT)
	n, m = np.shape(p)
	aom = 0  # amount of machines
	for i in range(len(mop)):
		aom += mop[i]
	# count = 0  # number of jobs that been completed, but it will be given value later
	complete_flag = np.zeros(n, dtype=int)  # a flag that tell you if the job done or not
	S = np.zeros_like(p)  # record the start of every job in every procedure
	sequence = []  # the sequence of precessing job
	# initialize every job's start time in each procedure
	# assuming that every job start at 0 and no conflict exist
	for i in range(n):
		for j in range(1, m):
			S[i, j] = S[i, j-1] + A[i, j-1]
	C = np.zeros_like(p)  # record the completing of every job in every procedure
	R = np.zeros(aom)  # record the release of every machine
	t = np.zeros(n)  # processing time of every job
	for i in range(n):
		t[i] = processing_time(i, A)
	min_work = np.argmin(t)

	# the min_work is processed as following
	C[min_work, 0] = A[min_work, 0]
	R[p[min_work, 0]] = C[min_work, 0]
	for j in range(1, m):
		C[min_work, j] = C[min_work, j-1] + A[min_work, j]
		R[p[min_work, j]] = C[min_work, j]
	complete_flag[min_work] = 1
	sequence.append(min_work)
	# the min_work done

	sec_work = parallel(min_work, p)
	if sec_work is not None:
		C[sec_work, 0] = A[sec_work, 0]
		R[p[sec_work, 0]] = C[sec_work, 0]
		for j in range(1, m):
			C[sec_work, j] = C[sec_work, j-1] + A[sec_work, j]
			R[p[sec_work, j]] = C[sec_work, j]
		complete_flag[sec_work] = 1
		count = 2
		sequence.append(sec_work)
	else:
		count = 1

	while count != n:
		conflict_time = {}
		C_max = {}  # the maximum complete time
		for i in range(n):
			if complete_flag[i] == 0:
				conflict_time[i] = conflict_check(i, p, S, R)
				C_max[i] = complete_max(i, conflict_time[i], A, C, sequence)

		k = sorted(C_max.items(), key=lambda item: item[1])[0][0]
		delay_time = conflict_time[k]

		# k_job is processed as following
		S[k, 0] = delay_time
		C[k, 0] = A[k, 0] + S[k, 0]
		R[p[k, 0]] = C[k, 0]
		for j in range(1, m):
			S[k, j] = C[k, j - 1]
			C[k, j] = S[k, j] + A[k, j]
			R[p[k, j]] = C[k, j]
		complete_flag[k] = 1
		count += 1
		sequence.append(k)

		# record the min_C_max
		min_C_max = C_max[k]
		if min_C_max == fg:
			display = True

	if display:
		print('The sequence of processing: \n', sequence)
		print('Start of every procedure: \n', S)
		print('End of every procedure: \n', C)
		print('processing time: \n', min_C_max)
		gatt(p, sequence, mop, S, A)

	return min_C_max
