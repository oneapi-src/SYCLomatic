CC = nvcc
TARGET = hello
WORK_DIR = $(shell pwd)

all: $(TARGET)

$(TARGET).o:$(TARGET).cu
	$(CC) -c -o $(TARGET).o $(TARGET).cu

$(TARGET): $(TARGET).o
	$(CC) -o $(WORK_DIR)/$(TARGET) $(TARGET).o

clean:
	rm -f $(TARGET) $(TARGET).o


   
