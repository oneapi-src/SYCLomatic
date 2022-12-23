//RUN: dpct -out-root %T/curand-usm %s --format-range=none --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-usm/curand-usm.dp.cpp --match-full-lines %s
//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <dpct/rng_utils.hpp>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

int main(){
  //CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
  //CHECK-NEXT:sycl::queue &q_ct1 = dev_ct1.default_queue();
  //CHECK:oneapi::mkl::rng::gaussian<double> distr_ct{{[0-9]+}}(123, 456);
  //CHECK-NEXT:oneapi::mkl::rng::gaussian<float> distr_ct{{[0-9]+}}(123, 456);
  //CHECK-NEXT:oneapi::mkl::rng::lognormal<double> distr_ct{{[0-9]+}}(123, 456, 0.0, 1.0);
  //CHECK-NEXT:oneapi::mkl::rng::lognormal<float> distr_ct{{[0-9]+}}(123, 456, 0.0, 1.0);
  //CHECK-NEXT:oneapi::mkl::rng::poisson<int32_t> distr_ct{{[0-9]+}}(123.456);
  //CHECK-NEXT:oneapi::mkl::rng::uniform<double> distr_ct{{[0-9]+}};
  //CHECK-NEXT:oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT:oneapi::mkl::rng::uniform_bits<uint32_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:oneapi::mkl::rng::uniform_bits<uint64_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:int s1;
  //CHECK-NEXT:int s2;
  //CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::philox4x32x10> rng;
  //CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(q_ct1, 1337ull);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:float *d_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data);
  curandGenerateUniform(rng, d_data, 100*100);


  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data), 0);
  s1 = curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data), 0);
  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data), 0);
  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data_d);
  curandGenerateUniformDouble(rng, d_data_d, 100*100);

  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data_d);
  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);

  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data_d);
  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, (uint32_t*)d_data_ui), 0);
  s1 = curandGenerate(rng, d_data_ui, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, (int32_t*)d_data_ui), 0);
  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, (uint64_t*)d_data_ull);
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, (uint64_t*)d_data_ull), 0)){}
  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if((oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, (uint64_t*)d_data_ull), 0)){}
  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:std::shared_ptr<oneapi::mkl::rng::sobol> rng2;
  //CHECK-NEXT:rng2 = std::make_shared<oneapi::mkl::rng::sobol>(q_ct1, 1111);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng2, 100*100, d_data);
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

#define M 100
#define N 200
  //CHECK:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng2, M * N, d_data);
  curandGenerateUniform(rng2, d_data, M * N);
#undef M
#undef N

  //CHECK:oneapi::mkl::rng::skip_ahead(*rng, 100);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::skip_ahead(*rng2, 200), 0);
  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  //CHECK:rng.reset();
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (rng.reset(), 0);
  curandDestroyGenerator(rng);
  s1 = curandDestroyGenerator(rng);

  //CHECK: dpct::queue_ptr stream;
  //CHECK-NEXT: rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(*stream, 1337ull);
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
//CHECK-NEXT:    rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(), 1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
//CHECK-NEXT:    */
//CHECK-NEXT:  }
//CHECK-NEXT:  ~A(){
//CHECK-NEXT:    rng.reset();
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  std::shared_ptr<oneapi::mkl::rng::sobol> rng;
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
//CHECK-NEXT:    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:    sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:    rng = std::make_shared<oneapi::mkl::rng::sobol>(q_ct1, 1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
//CHECK-NEXT:    */
//CHECK-NEXT:    karg1 = sycl::malloc_device<int>(32, q_ct1);
//CHECK-NEXT:  }
//CHECK-NEXT:  ~B(){
//CHECK-NEXT:    rng.reset();
//CHECK-NEXT:    sycl::free(karg1, dpct::get_default_queue());
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  std::shared_ptr<oneapi::mkl::rng::sobol> rng;
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

//CHECK:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::philox4x32x10> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(dpct::get_default_queue(), 1337ull);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed because this call is redundant in SYCL.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::sobol> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(), 1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
//CHECK-NEXT:*/
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
//CHECK:oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::philox4x32x10> rng;
//CHECK-NEXT:curandErrCheck((rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(dpct::get_default_queue(), 1337ull), 0));
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was replaced with 0 because this call is redundant in SYCL.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck(0);
//CHECK-NEXT:float *d_data;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck((oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data), 0));
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
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
//CHECK:/*
//CHECK-NEXT:DPCT1033:{{[0-9]+}}: Migrated code uses a basic Sobol generator. Initialize oneapi::mkl::rng::sobol generator with user-defined direction numbers to use it as Scrambled Sobol generator.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::sobol> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(), 1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

//CHECK:int bar5() try {
//CHECK-NEXT:  oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:  float *d_data;
//CHECK-NEXT:  std::shared_ptr<oneapi::mkl::rng::sobol> rng2;
//CHECK-NEXT:  rng2 = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(), 1111);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed because this call is redundant in SYCL.
//CHECK-NEXT:  */
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
//CHECK-NEXT:  */
//CHECK-NEXT:  return (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng2, 100*100, d_data), 0);
//CHECK-NEXT:}
int bar5(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar6(float *x_gpu, size_t n) {
  //CHECK: oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT: static std::shared_ptr<oneapi::mkl::rng::philox4x32x10> gen[16];
  static curandGenerator_t gen[16];
  static int init[16] = {0};
  int i = 0;
  if(!init[i]) {
    //CHECK: gen[i] = std::make_shared<oneapi::mkl::rng::philox4x32x10>(dpct::get_default_queue(), 1234);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed because this call is redundant in SYCL.
    //CHECK-NEXT: */
    curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen[i], 1234);
    init[i] = 1;
  }
  //CHECK: oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *(gen[i]), n, x_gpu);
  curandGenerateUniform(gen[i], x_gpu, n);
}

