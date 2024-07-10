CC        = nvcc
ARCH      = sm_60

program = all 

$(program): main.cpp test 
	$(CC) -c main.cpp test.o -o $@ $(LDFLAGS)


test: 
	$(CC) -c kernels/test.cu -o test.o -arch=all -arch=native -arch=all-major --gpu-architecture=all --gpu-architecture=native --gpu-architecture=all-major

clean:
	rm -rf *.o all compile_commands.json

