// RUN: dpct --format-range=none -out-root %T/atomic_user_defined %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/atomic_user_defined/atomic_user_defined.dp.cpp
#include <stdio.h>
#include <device_atomic_functions.h>

class A{
public:
  __device__ int atomicAdd(char * address, int value)
  {
    return ::atomicAdd((unsigned int*)address, value);
  }
};

namespace test{
  __device__ int atomicAdd(char * address, int value)
  {
    return ::atomicAdd((unsigned int*)address, value);
  }
}

__global__ void ker1(char *count)
{
  int n = 1, x = 0;
  // CHECK: x = test::atomicAdd(&count[0], n);
  x = test::atomicAdd(&count[0], n);
}

__global__ void ker2(char *count)
{
  int n = 1, x = 0;
  using test::atomicAdd;
  // CHECK: x = atomicAdd(&count[0], n);
  x = atomicAdd(&count[0], n);
}

__global__ void ker3(char *count)
{
  int n = 1, x = 0;
  using namespace test;
  // CHECK: x = atomicAdd(&count[0], n);
  x = atomicAdd(&count[0], n);
}

__global__ void ker4(char *count)
{
  int n = 1, x = 0;
  A a;
  // CHECK: x = a.atomicAdd(&count[0], n);
  x = a.atomicAdd(&count[0], n);
}

__global__ void ker5(unsigned int *count)
{
  int n = 1, x = 0;
  // CHECK: x = dpct::atomic_fetch_add(count, n);
  x = atomicAdd(count, n);
}

int main()
{
  char hitCount[1];
  char *hitCount_d;

  hitCount[0]=1;
  cudaMalloc((void **)&hitCount_d,1*sizeof(char));

  cudaMemcpy(&hitCount_d[0],&hitCount[0],1*sizeof(char),cudaMemcpyHostToDevice);

  ker1<<<1,4>>>(hitCount_d);

  cudaMemcpy(&hitCount[0],&hitCount_d[0],1*sizeof(char),cudaMemcpyDeviceToHost);

  printf("On HOST, count is %c\n",hitCount[0]);

  return 0;
}
