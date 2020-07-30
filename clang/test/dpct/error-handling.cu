// RUN: cat %s > %T/error-handling.cu
// RUN: cd %T
// RUN: dpct --usm-level=none -out-root %T error-handling.cu --cuda-include-path="%cuda-path/include" -- -w -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck error-handling.cu --match-full-lines --input-file %T/error-handling.dp.cpp

#include <stdexcept>
#include <cublas.h>
#include <vector>
int printf(const char *s, ...);
int fprintf(int, const char *s, ...);

// CHECK:void test_simple_ifs() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_simple_ifs() {
  cudaError_t err;
  if (err != cudaSuccess) {
  }
  if (err) {
  }
  if (err != 0) {
  }
  if (0 != err) {
  }
  if (cudaSuccess != err) {
  }
  if (err != cudaSuccess) {
  }
}

// CHECK:void test_simple_ifs_const() {
// CHECK-NEXT:  const int err = 0;
// CHECK-NEXT:}
void test_simple_ifs_const() {
  const cudaError_t err = cudaSuccess;
  if (err != cudaSuccess) {
  }
  if (err) {
  }
  if (err != 0) {
  }
  if (0 != err) {
  }
  if (cudaSuccess != err) {
  }
  if (err != cudaSuccess) {
  }
}

// CHECK:void test_typedef() {
// CHECK-NEXT:  typedef int someError_t;
// CHECK-NEXT:  someError_t err;
// CHECK-NEXT:}
void test_typedef()  {
  typedef cudaError_t someError_t;
  someError_t err;
  if (err != cudaSuccess) {
  }
  if (0 != err) {
  }
}

// CHECK:void test_no_braces() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_no_braces() {
  cudaError_t err;
  if (err != cudaSuccess)
    printf("error!\n");
}

// CHECK:void test_unrelated_then() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:  int i = 0;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:   if (err != 0) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    ++i;
// CHECK-NEXT:  }
// CHECK-NEXT:}

void test_unrelated_then() {
  cudaError_t err;
  int i = 0;
  if (err != cudaSuccess) {
    ++i;
  }
}

// CHECK:void test_CUDA_SUCCESS() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_CUDA_SUCCESS() {
  cudaError_t err;
  if (err != CUDA_SUCCESS) {
    printf("error!\n");
  }
}

// CHECK:void test_CUDA_SUCCESS_empty() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_CUDA_SUCCESS_empty() {
  cudaError_t err;
  if (err != CUDA_SUCCESS) {
  }
}

// CHECK:void test_CUDA_SUCCESS_CUresult() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_CUDA_SUCCESS_CUresult() {
  CUresult err;
  if (err != CUDA_SUCCESS) {
    printf("error!\n");
  }
}

// CHECK:void test_CUDA_SUCCESS_empty_CUresult() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void test_CUDA_SUCCESS_empty_CUresult() {
  CUresult err;
  if (err != CUDA_SUCCESS) {
  }
}

// CHECK:void test_other_enum() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:  if (err != {{[0-9]+}}) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_other_enum() {
  cudaError_t err;
  if (err != cudaErrorLaunchFailure) {
    printf("error!\n");
  }
}

// CHECK:void test_assignment() try {
// CHECK-NEXT:  int err;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err = (dpct::dpct_malloc(0, 0), 0)) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void test_assignment() {
  cudaError_t err;
  if (err = cudaMalloc(0, 0)) {
    printf("error!\n");
  }
}

// CHECK:void test_1(int err, int arg) {
// CHECK-NEXT:  if (err == 0 && arg) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_1(cudaError_t err, int arg) {
  if (err == cudaSuccess && arg) {
  }
}

// CHECK:void test_12(int err, int arg) {
// CHECK-NEXT:  if (err) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:  {{ +}}
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_12(cudaError_t err, int arg) {
  if (err) {
  } else {
    
  }
}

// CHECK:void test_13(int err, int arg) {
// CHECK-NEXT:}
void test_13(cudaError_t err, int arg) {
  if (err) {
    printf("error!\n");
  }
}

// CHECK:void test_14(int err, int arg) {
// CHECK-NEXT:  if (arg == 1) {
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-NEXT  if (arg != 0) {
// CHECK-NEXT    return;
// CHECK-NEXT  }
// CHECK-NEXT  if (arg) {
// CHECK-NEXT    return;
// CHECK-NEXT  }
// CHECK-NEXT}
void test_14(cudaError_t err, int arg) {
  if (arg == 1) {
    return;
  }
  if (arg != 0) {
    return;
  }
  if (arg) {
    return;
  }
}

// CHECK:void test_15(int err, int arg) try {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if ((dpct::dpct_malloc(0, 0), 0)) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void test_15(cudaError_t err, int arg) {
  if (cudaMalloc(0, 0)) {
  }
}

// CHECK:void test_16(int err, int arg) {
// CHECK-NEXT:  if (err) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  } else {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_16(cudaError_t err, int arg) {
  if (err) {
    printf("error!\n");
    exit(1);
  } else {
    
  }
}

// CHECK:void test_17(int err, int arg)  try {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (!(dpct::dpct_malloc(0, 0), 0)) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void test_17(cudaError_t err, int arg) {
  if (!cudaMalloc(0, 0)) {
  } else {
    printf("error!\n");
    exit(1);
  }
}

// CHECK:void test_18(int err, int arg) {
// CHECK-NEXT:  if (err)
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  else
// CHECK-NEXT:    printf("success!\n");
// CHECK-NEXT:}
void test_18(cudaError_t err, int arg) {
  if (err)
    printf("error!\n");
  else
    printf("success!\n");
}

// CHECK:void test_19(int err, int arg) {
// CHECK-NEXT:  if (err && arg) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_19(cudaError_t err, int arg) {
  if (err && arg) {
  } else {
  }
}

// CHECK:void test_compare_to_3(int err, int arg) {
// CHECK-NEXT:  if (err != 3) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_compare_to_3(cudaError_t err, int arg) {
  if (err != 3) {
  }
}

// CHECK:void test_21(const int &err, int arg) {
// CHECK-NEXT:}
void test_21(const cudaError_t& err, int arg) {
  if (err != 0) {
  }
}

// CHECK:void test_no_side_effects(int err, int arg) {
// CHECK-NEXT:}
void test_no_side_effects(cudaError_t err, int arg) {
  if (err)
    printf("efef");
  if (err)
    fprintf(0, "efef");
  if (err)
    exit(1);
  if (err != cudaSuccess) {
    printf("error!\n");
    exit(1);
  }
}

// CHECK:void test_side_effects(int err, int arg, int x, int y, int z) {
// CHECK:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err != 0) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    x = printf("fmt string");
// CHECK-NEXT:}

void test_side_effects(cudaError_t err, int arg, int x, int y, int z) {
  if (err)
    printf("efef %i", malloc(0x100));
  if (err)
    malloc(0x100);
  if (err != cudaSuccess) {
    malloc(0x100);
    printf("error!\n");
    exit(1);
  }
  if (err)
    x = printf("fmt string");
  if (err)
    printf("fmt string %d", y + z);
}

// CHECK:void specialize_ifs() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:}
void specialize_ifs() {
  cudaError_t err;
  if (err == cudaErrorAssert) {
    printf("efef");
  }
  if (err == 255) {
  }
  if (err == 1) {
  }
  if (666 == err) {
  }
  if (cudaErrorAssert == err) {
  }
}

// CHECK:void specialize_ifs_negative() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:  if (err == 0) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to
// CHECK-NEXT:rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == {{[0-9]+}}) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to
// CHECK-NEXT:rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 255) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to
// CHECK-NEXT:rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 1) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to
// CHECK-NEXT:rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (666 == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to
// CHECK-NEXT:rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if ({{[0-9]+}} == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:}
void specialize_ifs_negative() {
  cudaError_t err;
  if (err == cudaSuccess) {
    printf("efef");
  }
  if (err == cudaErrorAssert) {
    printf("efef");
    malloc(0x100);
  }
  if (err == 255) {
    malloc(0x100);
  }
  if (err == 1) {
    malloc(0x100);
  }
  if (666 == err) {
    malloc(0x100);
  }
  if (cudaErrorAssert == err) {
    malloc(0x100);
  }
}

// CHECK: void foo1() try {
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   if ((dpct::dpct_malloc(0, 0), 0)) {
// CHECK-NEXT:     printf("efef");
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void foo1(){
  if(cudaMalloc(0, 0)){
    printf("efef");
  }
}


// CHECK: void foo2() try {
// CHECK-NEXT:   size_t size = 1234567 * sizeof(float);
// CHECK-NEXT:   float *h_A = (float *)malloc(size);
// CHECK-NEXT:   float *d_A = NULL;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   while ((dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device), 0)) {
// CHECK-NEXT:     printf("efef");
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void foo2(){
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  while(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice)){
    printf("efef");
  }
}


// CHECK: void foo3() try {
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   for (; (dpct::dpct_malloc(0, 0), 0);) {
// CHECK-NEXT:     printf("efef");
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void foo3(){
  for(;cudaMalloc(0, 0);){
    printf("efef");
  }
}



// CHECK: void foo4() try {
// CHECK-NEXT:   do{
// CHECK-NEXT:     printf("efef");
// CHECK-NEXT:     /*
// CHECK-NEXT:     DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted.
// CHECK-NEXT:     You may need to rewrite this code.
// CHECK-NEXT:     */
// CHECK-NEXT:   } while ((dpct::dpct_malloc(0, 0), 0));
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT:   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:   std::exit(1);
// CHECK-NEXT: }
void foo4(){
  do{
    printf("efef");
  } while(cudaMalloc(0, 0));
}


// CHECK: void foo5(){
// CHECK-NEXT:   int res;
// CHECK-NEXT:   {
// CHECK-NEXT:   auto ct_0_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(0);
// CHECK-NEXT:   sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
// CHECK-NEXT:   oneapi::mkl::blas::iamax(*dpct::get_current_device().get_saved_queue(), 10,
// CHECK-NEXT:                    ct_0_buf_ct{{[0-9]+}}, 0, res_temp_buf_ct{{[0-9]+}});
// CHECK-NEXT:   res = res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access::mode::read>()[0];
// CHECK-NEXT:   }
// CHECK-NEXT: }
void foo5(){
  int res = cublasIsamax(10, 0, 0);
}

// CHECK: void foo6(){
// CHECK-NEXT:   int a;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   a = (dpct::dpct_malloc(0, 0), 0);
// CHECK-NEXT: }
void foo6(){
  int a;
  a = cudaMalloc(0, 0);
}

// CHECK: void foo7(){
// CHECK-NEXT:   dpct::dpct_malloc(0, 0);
// CHECK-NEXT:   int a = printf("a");
// CHECK-NEXT:   if(printf("a")){}
// CHECK-NEXT:   sycl::event start;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0, because this
// CHECK-NEXT:   call is redundant in DPC++.
// CHECK-NEXT:   */
// CHECK-NEXT:   int b = 0;
// CHECK-NEXT: }
void foo7(){
  cudaMalloc(0, 0);
  int a = printf("a");
  if(printf("a")){}
  cudaEvent_t start;
  int b = cudaEventCreate(&start);
}

// CHECK: void foo8() try {
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   int a = (dpct::dpct_malloc(0, 0), 0);
// CHECK-NEXT:   if(a) printf("a");
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT: std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT: std::exit(1);
// CHECK-NEXT: }
void foo8(){
  int a = cudaMalloc(0, 0);
  if(a) printf("a");
}

// CHECK: void foo9() try {
// CHECK-NEXT:   int a;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
// CHECK-NEXT:   may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   a = (dpct::dpct_malloc(0, 0), 0);
// CHECK-NEXT:   if(a) printf("a");
// CHECK-NEXT: }
// CHECK-NEXT: catch (sycl::exception const &exc) {
// CHECK-NEXT: std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT: std::exit(1);
// CHECK-NEXT: }
void foo9(){
  int a;
  a = cudaMalloc(0, 0);
  if(a) printf("a");
}

//CHECK: int foo10() try {
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:   may need to rewrite this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   return (dpct::dpct_malloc(0, 0), 0);
//CHECK-NEXT: }
//CHECK-NEXT: catch (sycl::exception const &exc) {
//CHECK-NEXT: std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
//CHECK-NEXT: std::exit(1);
//CHECK-NEXT: }
int foo10(){
  return cudaMalloc(0, 0);
}

//CHECK: int foo11(){
//CHECK-NEXT: while(true){
//CHECK-NEXT:   dpct::dpct_malloc(0, 0);
//CHECK-NEXT: }
//CHECK-NEXT: }
int foo11(){
  while(true){
    cudaMalloc(0, 0);
  }
}

//CHECK: void foo12() try {
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:   may need to rewrite this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   switch ((dpct::dpct_malloc(0, 0), 0)) {
//CHECK-NEXT:     case 0:
//CHECK-NEXT:       break;
//CHECK-NEXT:     case 1:
//CHECK-NEXT:       break;
//CHECK-NEXT:     default:
//CHECK-NEXT:     ;
//CHECK-NEXT:   }
//CHECK-NEXT: }
//CHECK-NEXT: catch (sycl::exception const &exc) {
//CHECK-NEXT: std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:             << ", line:" << __LINE__ << std::endl;
//CHECK-NEXT: std::exit(1);
//CHECK-NEXT: }
void foo12(){
  switch(cudaMalloc(0, 0)){
    case 0:
      break;
    case 1:
      break;
    default:
    ;
  }
}


// CHECK: class ClassA
// CHECK-NEXT: {
// CHECK-NEXT:   public:
// CHECK-NEXT:   std::vector<int> V;
// CHECK-NEXT:   ClassA() : V() {}
// CHECK-NEXT:   ClassA(int b) try {
// CHECK-NEXT:     /*
// CHECK-NEXT:     DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted.
// CHECK-NEXT:     You may need to rewrite this code.
// CHECK-NEXT:     */
// CHECK-NEXT:     int a = (dpct::dpct_malloc(0, 0), 0);
// CHECK-NEXT:     if(a){printf("a");}
// CHECK-NEXT:   }
// CHECK-NEXT:   catch (sycl::exception const &exc) {
// CHECK-NEXT:     std::cerr << exc.what() << "Exception caught at file:" << __FILE__
// CHECK-NEXT:               << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:     std::exit(1);  
// CHECK-NEXT:   }
// CHECK-NEXT: };
class ClassA
{
  public:
  std::vector<int> V;
  ClassA() : V() {}
  ClassA(int b)
  {
    int a = cudaMalloc(0, 0);
    if(a){printf("a");}
  }
};

// CHECK: __dpct_inline__ int foo13(int error, const char *filename, int line) {
// CHECK-NEXT:  int error_1 = 0;
// CHECK-NEXT:  return error_1;
// CHECK-NEXT: }
__host__ __device__ __forceinline__ cudaError_t foo13(cudaError_t error,
                                                    const char *filename,
                                                    int line) {
  cudaError_t error_1 = cudaSuccess;
  return error_1;
}

// CHECK: __dpct_inline__ int *foo14(int error, const char *filename, int line) {
// CHECK-NEXT:  return &error;
// CHECK-NEXT: }
__host__ __device__ __forceinline__ cudaError_t *foo14(cudaError_t error,
                                                    const char *filename,
                                                    int line) {
  return &error;
}


//CHECK: int foo15(){
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error
//CHECK-NEXT:   codes. The call was replaced with 0. You need to rewrite this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   return 0;
//CHECK-NEXT: }
int foo15(){
  return cudaGetLastError();
}

//CHECK: int foo16(){
//CHECK-NEXT:   int *a;
//CHECK-NEXT:   return [&]() {
//CHECK-NEXT:     /*
//CHECK-NEXT:     DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted.
//CHECK-NEXT:     You may need to rewrite this code.
//CHECK-NEXT:     */
//CHECK-NEXT:     try {
//CHECK-NEXT:       return (dpct::dpct_malloc((void **)&a, 0), 0);
//CHECK-NEXT:     }
//CHECK-NEXT:     catch (sycl::exception const &exc) {
//CHECK-NEXT:       std::cerr << exc.what() << "Exception caught at file:" << __FILE__
//CHECK-NEXT:                 << ", line:" << __LINE__ << std::endl;
//CHECK-NEXT:       std::exit(1);
//CHECK-NEXT:     }
//CHECK-NEXT:   }();
//CHECK-NEXT: }
int foo16(){
  int *a;
  return [&](){
    return cudaMalloc((void**)&a,0);
    }();
}

