CFLAG = -O3 -g -Wall -fopenmp

all: 
	nvcc driver.cc winograd.cc winograd_cuda.cu -o winograd -lcublas

clean:
	rm -f winograd