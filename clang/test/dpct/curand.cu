// CHECKME
// RUN: cat %s > %T/curand.cu
// RUN: cd %T
//RUN: dpct -out-root %T/curand curand.cu --usm-level=none --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand/curand.dp.cpp --match-full-lines curand.cu
//CHECK:// CHECKME
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
  //CHECK-NEXT:rng =
  //CHECK-NEXT:    dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
  //CHECK-NEXT:rng->set_seed(1337ull);
  //CHECK-NEXT:float *d_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  //CHECK:rng->generate_uniform(d_data, 100 * 100);
  curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng->generate_uniform(d_data, 100 * 100), 0);
  s1 = curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng->generate_lognormal(d_data, 100 * 100, 123, 456), 0);
  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng->generate_gaussian(d_data, 100 * 100, 123, 456), 0);
  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  //CHECK:rng->generate_uniform(d_data_d, 100 * 100);
  //CHECK-NEXT:rng->generate_lognormal(d_data_d, 100 * 100, 123, 456);
  //CHECK-NEXT:rng->generate_gaussian(d_data_d, 100 * 100, 123, 456);
  curandGenerateUniformDouble(rng, d_data_d, 100*100);
  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);
  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng->generate_uniform_bits(d_data_ui, 100 * 100), 0);
  s1 = curandGenerate(rng, d_data_ui, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng->generate_poisson(d_data_ui, 100 * 100, 123.456), 0);
  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  //CHECK:rng->generate_uniform_bits(d_data_ull, 100 * 100);
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if (s1 = (rng->generate_uniform_bits(d_data_ull, 100 * 100), 0)) {}
  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if ((rng->generate_uniform_bits(d_data_ull, 100 * 100), 0)) {}
  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:dpct::rng::host_rng_ptr rng2;
  //CHECK-NEXT:rng2 = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
  //CHECK-NEXT:rng2->set_dimensions(1111);
  //CHECK-NEXT:rng2->generate_uniform(d_data, 100 * 100);
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

  //CHECK:rng->skip_ahead(100);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng2->skip_ahead(200), 0);
  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  //CHECK:rng.reset();
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng.reset(), 0);
  curandDestroyGenerator(rng);
  s1 = curandDestroyGenerator(rng);
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
//CHECK-NEXT:private:
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



void bar1(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to
//CHECK-NEXT:adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:rng =
//CHECK-NEXT:    dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
//CHECK-NEXT:rng->set_seed(1337ull);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to
//CHECK-NEXT:adjust the code.
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
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck((rng = dpct::rng::create_host_rng(
//CHECK-NEXT:         dpct::rng::random_engine_type::philox4x32x10),
//CHECK-NEXT:     0));
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck((rng->set_seed(1337ull), 0));
//CHECK-NEXT:float *d_data;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck((rng->generate_uniform(d_data, 100 * 100), 0));
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck((rng.reset(), 0));
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
//CHECK-NEXT:DPCT1033:{{[0-9]+}}: Migrated code uses a basic Sobol generator. Initialize
//CHECK-NEXT:oneapi::mkl::rng::sobol generator with user-defined direction numbers to use
//CHECK-NEXT:it as Scrambled Sobol generator.
//CHECK-NEXT:*/
//CHECK-NEXT:rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:rng->set_dimensions(1243);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

void bar5(){
//CHECK:dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1028:{{[0-9]+}}: The curandCreateGenerator was not migrated because parameter
//CHECK-NEXT:(curandRngType)101 is unsupported.
//CHECK-NEXT:*/
//CHECK-NEXT:curandCreateGenerator(&rng, (curandRngType)101);
//CHECK-NEXT:rng->set_seed(1337ull);
  curandGenerator_t rng;
  curandCreateGenerator(&rng, (curandRngType)101);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}

//CHECK:int bar6() try {
//CHECK-NEXT:  float *d_data;
//CHECK-NEXT:  dpct::rng::host_rng_ptr rng2;
//CHECK-NEXT:  rng2 = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);
//CHECK-NEXT:  rng2->set_dimensions(1111);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
//CHECK-NEXT:  may need to rewrite this code.
//CHECK-NEXT:  */
//CHECK-NEXT:  return (rng2->generate_uniform(d_data, 100 * 100), 0);
//CHECK-NEXT:}
int bar6(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar7() {
  curandGenerator_t rng;
  //CHECK:rng =
  //CHECK:    dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
  curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
