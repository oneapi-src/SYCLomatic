TARGET = hello

all: 
	CPATH=;nvcc -o $(TARGET) *.cu
	CPATH=; nvcc -o $(TARGET) *.cu
	/usr/local/cuda-12.0/bin/nvcc -o $(TARGET) *.cu
	@echo " " nvcc -o $(TARGET) *.cu
	nvcc -o $(TARGET) *.cu

clean:
	rm -f $(TARGET)


   
