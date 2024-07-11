// RUN: echo 0
#include "test.cuh"
__global__ void test_print(){
	  printf("Hello World!\n");
}

void wrap_test_print() {
	  test_print<<<1, 1>>>();
	    return;
}

