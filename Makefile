CFLAG = -O3 -g -Wall -fopenmp

all: 
	nvcc -use_fast_math -Xptxas -O3 driver.cc winograd.cc winograd_cuda.cu -o winograd -lcublas

clean:
	rm -f winograd
	rm -f -r slurm-output
	rm -f -r slurm-error
	rm -f -r vtune