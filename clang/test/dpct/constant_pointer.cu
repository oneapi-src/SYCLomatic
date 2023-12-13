// RUN: dpct --format-range=none --usm-level=none -out-root %T/constant_pointer %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/constant_pointer/constant_pointer.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/constant_pointer/constant_pointer.dp.cpp -o %T/constant_pointer/constant_pointer.dp.o %}

// CHECK: static dpct::global_memory<int, 1> schsfirst;
static __constant__ const int *schsfirst;
// CHECK: static dpct::global_memory<sycl::double2, 1> zm;
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
  //CHECK: schsfirst.assign(schsfirstD, numschH * sizeof(int));
  cudaMemcpyToSymbol(schsfirst, &schsfirstD, sizeof(void *));

  cudaMalloc(&zmD, numschH * sizeof(double2));
  //CHECK: zm.assign(zmD, numschH * sizeof(sycl::double2));
  cudaMemcpyToSymbol(zm, &zmD, sizeof(void *));

  gpuMain2<<<1, 1>>>();

  cudaFree(schsfirstD);
  cudaFree(zmD);
};

int main() {
  init();
}


