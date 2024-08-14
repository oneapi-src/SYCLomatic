# Makefile for compiling vector_add.cu

# Specify the compiler
NVCC = nvcc

# Specify the target file
TARGET = vector_add

# Specify the source files
SOURCES = vector_add.cu

# Specify the build flags
NVCC_FLAGS = -O2 -arch=sm_50

# Specify the output file
OUT = -o $(TARGET)

# Default make command
all: $(TARGET)

# Rule for building the target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) $(OUT)

# Clean command to remove compiled files
clean:
	rm -f $(TARGET)
