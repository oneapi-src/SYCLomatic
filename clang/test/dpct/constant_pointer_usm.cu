// RUN: dpct --format-range=none --usm-level=restricted -out-root %T/constant_pointer_usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/constant_pointer_usm/constant_pointer_usm.dp.cpp

// CHECK: static dpct::constant_memory<const int *, 0> schsfirst;
static __constant__ const int *schsfirst;
// CHECK: static dpct::constant_memory<const sycl::double2 *, 0> zm;
static __constant__ const double2 *zm;

static int *schsfirstD;
static double2 *zmD;

static __global__ void gpuMain2() {
  const int s0 = threadIdx.x;
  const int sch = blockIdx.x;
  const int s = schsfirst[sch] + s0;
}

void init() {
  int numschH = 100;

  cudaMalloc(&schsfirstD, numschH * sizeof(int));
  //CHECK: q_ct1.memcpy(schsfirst.get_ptr(), &schsfirstD, sizeof(void *)).wait();
  cudaMemcpyToSymbol(schsfirst, &schsfirstD, sizeof(void *));

  cudaMalloc(&zmD, numschH * sizeof(double2));
  //CHECK: q_ct1.memcpy(zm.get_ptr(), &zmD, sizeof(void *));
  cudaMemcpyToSymbol(zm, &zmD, sizeof(void *));

  gpuMain2<<<1, 1>>>();

  cudaFree(schsfirstD);
  cudaFree(zmD);
};

int main() {
  init();
}


