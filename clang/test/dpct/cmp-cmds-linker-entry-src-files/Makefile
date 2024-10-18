#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
EXTRA_CFLAGS = --generate-dependencies-with-compile --generate-nonsystem-dependencies-with-compile --allow-unsupported-compiler -allow-unsupported-compiler
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xfatbin=-compress-all -Xfatbin -compress-all -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): one.o two.o three.cpp four.c 
	$(CC) $(CFLAGS) one.o two.o three.cpp four.c -o $@ $(LDFLAGS)

%.o: %.cu 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o  *.bmp *.bin

run: $(program)
	./$(program)
