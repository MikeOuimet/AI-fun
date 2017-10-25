import numpy as np
cimport numpy as np


DTYPE = np.int
ctypedef np.int_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def example(int n):
	cdef int i = 0
	cdef np.ndarray[DTYPE_t, ndim = 2] data = np.empty(shape = (1, n), dtype = DTYPE)
	while True:
		for i in range(n):
			data[0, i] = i
		else:
			break