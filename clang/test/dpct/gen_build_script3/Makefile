TARGET = hello
COMPILER = nvcc

CHECK_DEFAULT_INSTALL_PATH = $(shell ls /usr/local | grep -E "^cuda-[0-9]+\.[0-9]+" > /dev/null 2>&1 && echo "available" || echo "not available")
CHECK_NVCC_FROM_PATH = $(shell nvcc --version >/dev/null 2>&1 && echo "available" || echo "not available")

ifneq ($(CHECK_NVCC_FROM_PATH), available)
    ifeq ($(CHECK_DEFAULT_INSTALL_PATH), available)
    COMPILER = echo "dump compiler"
    endif
endif

all: 
	CPATH=;$(COMPILER) -o $(TARGET) *.cu
	CPATH=; $(COMPILER) -o $(TARGET) *.cu
	/usr/local/cuda-12.0/bin/nvcc -o $(TARGET) *.cu
	@echo " " $(COMPILER) -o $(TARGET) *.cu
	$(COMPILER) -o $(TARGET) *.cu

clean:
	rm -f $(TARGET)
