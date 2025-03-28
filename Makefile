CFLAG = -O3 -g -Wall -fopenmp

all:
	nvcc driver.cc winograd.cc cuda_kernel.cu -o winograd -lcublas

clean:
	rm -f winograd