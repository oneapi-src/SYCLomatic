// RUN: echo
void foo(){
  int *dOut;
  //CHECK: CALL(dOut = sycl::malloc_device<int>(5, q_ct1));
  CALL(cudaMalloc((void **)&dOut, sizeof(int)*5));
  //CHECK: dOut = sycl::malloc_device<int>(5, q_ct1);
  cudaMalloc((void **)&dOut, sizeof(int)*5);
}