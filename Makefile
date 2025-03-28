CFLAG = -O3 -g -Wall -fopenmp

all: 
	nvcc driver.cc winograd.cc winograd_4x4_3x3.cu -o winograd -lcublas

clean:
	rm -f winograd