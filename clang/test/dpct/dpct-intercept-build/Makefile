CC        = nvcc
ARCH      = sm_60

program = all 

$(program): main.cpp test 
	$(CC) -c main.cpp test.o -o $@ $(LDFLAGS)


test: 
	$(CC) -c kernels/test.cu -o test.o 

clean:
	rm -rf *.o all compile_commands.json

