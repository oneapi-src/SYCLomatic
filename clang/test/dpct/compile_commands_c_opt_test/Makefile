all: one two three

one: one.cu
	$(CC) -O2 --use_fast_math -DKERNEL_USE_PROFILE=1 -gencode arch=compute_${USE_SM},code=sm_${USE_SM} -DRUN_ON_GPU=1 one.cu -o one -allow-unsupported-compiler

two: two.cu
	$(CC) -O2 --use_fast_math -DKERNEL_USE_PROFILE=1 -gencode arch=compute_${USE_SM},code=sm_${USE_SM} -DRUN_ON_GPU=1 two.cu

three: three.cu four.cu
	$(CC) -O2 --use_fast_math -DKERNEL_USE_PROFILE=1 -gencode arch=compute_${USE_SM},code=sm_${USE_SM} -DRUN_ON_GPU=1 three.cu four.cu