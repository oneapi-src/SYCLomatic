//RUN: dpct -out-root %T/curand-usm %s --format-range=none --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-usm/curand-usm.dp.cpp --match-full-lines %s
//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <dpct/rng_utils.hpp>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

int main(){
  //CHECK:int s1;
  //CHECK-NEXT:int s2;
  //CHECK-NEXT:dpct::rng::host_rng_ptr rng;
  //CHECK-NEXT:rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
  //CHECK-NEXT:rng->set_seed(1337ull);
  //CHECK-NEXT:float *d_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  //CHECK:rng->generate_uniform(d_data, 100*100);
  curandGenerateUniform(rng, d_data, 100*100);


  //CHECK:s1 = DPCT_CHECK_ERROR(rng->generate_uniform(d_data, 100*100));
  s1 = curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:s1 = DPCT_CHECK_ERROR(rng->generate_lognormal(d_data, 100*100, 123, 456));
  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  //CHECK:s1 = DPCT_CHECK_ERROR(rng->generate_gaussian(d_data, 100*100, 123, 456));
  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  //CHECK:rng->generate_uniform(d_data_d, 100*100);
  //CHECK-NEXT:rng->generate_lognormal(d_data_d, 100*100, 123, 456);
  //CHECK-NEXT:rng->generate_gaussian(d_data_d, 100*100, 123, 456);
  curandGenerateUniformDouble(rng, d_data_d, 100*100);
  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);
  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;
  //CHECK:s1 = DPCT_CHECK_ERROR(rng->generate_uniform_bits(d_data_ui, 100*100));
  s1 = curandGenerate(rng, d_data_ui, 100*100);

  //CHECK:s1 = DPCT_CHECK_ERROR(rng->generate_poisson(d_data_ui, 100*100, 123.456));
  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  //CHECK:rng->generate_uniform_bits(d_data_ull, 100*100);
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  //CHECK:if(s1 = DPCT_CHECK_ERROR(rng->generate_uniform_bits(d_data_ull, 100*100))){}
  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:if(DPCT_CHECK_ERROR(rng->generate_uniform_bits(d_data_ull, 100*100))){}
  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:dpct::rng::host_rng_ptr rng2;
  //CHECK-NEXT:rng2 = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
  //CHECK-NEXT:rng2->set_dimensions(1111);
  //CHECK-NEXT:rng2->generate_uniform(d_data, 100*100);
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

#define M 100
#define N 200
//CHECK:rng2->generate_uniform(d_data, M * N);
  curandGenerateUniform(rng2, d_data, M * N);
#undef M
#undef N

  //CHECK:rng->skip_ahead(100);
  //CHECK:s1 = DPCT_CHECK_ERROR(rng2->skip_ahead(200));
  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  //CHECK:rng.reset();
  //CHECK-NEXT:s1 = DPCT_CHECK_ERROR(rng.reset());
  curandDestroyGenerator(rng);
  s1 = curandDestroyGenerator(rng);

  //CHECK: dpct::queue_ptr stream;
  //CHECK-NEXT: rng->set_queue(stream);
  cudaStream_t stream;
  curandSetStream(rng, stream);
}

//CHECK:int foo1();
curandStatus_t foo1();
//CHECK:int foo2();
curandStatus foo2();

//CHECK:class A{
//CHECK-NEXT:public:
//CHECK-NEXT:  A(){
//CHECK-NEXT:    rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:    rng->set_dimensions(1243);
//CHECK-NEXT:  }
//CHECK-NEXT:  ~A(){
//CHECK-NEXT:    rng.reset();
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:};
class A{
public:
  A(){
    curandCreateGenerator(&rng, CURAND_RNG_QUASI_DEFAULT);
    curandSetQuasiRandomGeneratorDimensions(rng, 1243);
  }
  ~A(){
    curandDestroyGenerator(rng);
  }
private:
  curandGenerator_t rng;
};

//CHECK:class B{
//CHECK-NEXT:public:
//CHECK-NEXT:  B(){
//CHECK-NEXT:    rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:    rng->set_dimensions(1243);
//CHECK-NEXT:    karg1 = sycl::malloc_device<int>(32, dpct::get_default_queue());
//CHECK-NEXT:  }
//CHECK-NEXT:  ~B(){
//CHECK-NEXT:    rng.reset();
//CHECK-NEXT:    sycl::free(karg1, dpct::get_default_queue());
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:  int *karg1;
//CHECK-NEXT:};
class B{
public:
  B(){
    curandCreateGenerator(&rng, CURAND_RNG_QUASI_DEFAULT);
    curandSetQuasiRandomGeneratorDimensions(rng, 1243);
    cudaMalloc(&karg1, 32 * sizeof(int));
  }
  ~B(){
    curandDestroyGenerator(rng);
    cudaFree(karg1);
  }
private:
  curandGenerator_t rng;
  int *karg1;
};

void bar1(){

//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
//CHECK-NEXT:rng->set_seed(1337ull);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:rng->set_dimensions(1243);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
//CHECK:if (stat != 0) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

void bar3(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:curandErrCheck(DPCT_CHECK_ERROR(rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10)));
//CHECK-NEXT:curandErrCheck(DPCT_CHECK_ERROR(rng->set_seed(1337ull)));
//CHECK-NEXT:float *d_data;
//CHECK-NEXT:curandErrCheck(DPCT_CHECK_ERROR(rng->generate_uniform(d_data, 100*100)));
//CHECK-NEXT:curandErrCheck(DPCT_CHECK_ERROR(rng.reset()));
  curandGenerator_t rng;
  curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
  float *d_data;
  curandErrCheck(curandGenerateUniform(rng, d_data, 100*100));
  curandErrCheck(curandDestroyGenerator(rng));
}

void bar4(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1033:{{[0-9]+}}: Migrated code uses a basic Sobol generator. Initialize oneapi::mkl::rng::sobol generator with user-defined direction numbers to use it as Scrambled Sobol generator.
//CHECK-NEXT:*/
//CHECK-NEXT:rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:rng->set_dimensions(1243);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

//CHECK:int bar5() try {
//CHECK-NEXT:  float *d_data;
//CHECK-NEXT:  dpct::rng::host_rng_ptr rng2;
//CHECK-NEXT:  rng2 = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:  rng2->set_dimensions(1111);
//CHECK-NEXT:  return DPCT_CHECK_ERROR(rng2->generate_uniform(d_data, 100*100));
//CHECK-NEXT:}
int bar5(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar6(float *x_gpu, size_t n) {
  //CHECK: static dpct::rng::host_rng_ptr gen[16];
  static curandGenerator_t gen[16];
  static int init[16] = {0};
  int i = 0;
  if(!init[i]) {
    //CHECK: *(&gen[i]) = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
    //CHECK-NEXT: gen[i]->set_seed(1234);
    curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen[i], 1234);
    init[i] = 1;
  }
  //CHECK: gen[i]->generate_uniform(x_gpu, n);
  curandGenerateUniform(gen[i], x_gpu, n);
}

