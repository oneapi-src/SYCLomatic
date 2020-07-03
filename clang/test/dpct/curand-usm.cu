//RUN: dpct -out-root %T %s --format-range=none --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-usm.dp.cpp --match-full-lines %s
//CHECK:#include <CL/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <mkl_rng_sycl.hpp>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

int main(){
  //CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
  //CHECK-NEXT:sycl::queue &q_ct1 = dev_ct1.default_queue();
  //CHECK:int s1;
  //CHECK-NEXT:int s2;
  //CHECK-NEXT:mkl::rng::philox4x32x10 rng(q_ct1, 1337ull);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:float *d_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  //CHECK:mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data);
  curandGenerateUniform(rng, d_data, 100*100);


  //CHECK:mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data), 0);
  s1 = curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:mkl::rng::lognormal<float> distr_ct{{[0-9]+}}(123, 456, 0.0, 1.0);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data), 0);
  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  //CHECK:mkl::rng::gaussian<float> distr_ct{{[0-9]+}}(123, 456);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data), 0);
  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  //CHECK:mkl::rng::uniform<double> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data_d);
  curandGenerateUniformDouble(rng, d_data_d, 100*100);

  //CHECK:mkl::rng::lognormal<double> distr_ct{{[0-9]+}}(123, 456, 0.0, 1.0);
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data_d);
  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);

  //CHECK:mkl::rng::gaussian<double> distr_ct{{[0-9]+}}(123, 456);
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data_d);
  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;
  //CHECK:mkl::rng::uniform_bits<uint32_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, (uint32_t*)d_data_ui), 0);
  s1 = curandGenerate(rng, d_data_ui, 100*100);

  //CHECK:mkl::rng::poisson<int32_t> distr_ct{{[0-9]+}}(123.456);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, (int32_t*)d_data_ui), 0);
  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  //CHECK:mkl::rng::uniform_bits<uint64_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, (uint64_t*)d_data_ull);
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  //CHECK:mkl::rng::uniform_bits<uint64_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, (uint64_t*)d_data_ull);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK:if(s1 = 0){}
  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:mkl::rng::uniform_bits<uint64_t> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, (uint64_t*)d_data_ull);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(0){}
  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:mkl::rng::sobol rng2(q_ct1, 1111);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng2, 100*100, d_data);
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

  //CHECK:mkl::rng::skip_ahead(rng, 100);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::skip_ahead(rng2, 200), 0);
  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  //CHECK:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandDestroyGenerator was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandDestroyGenerator was replaced with 0, because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = 0;
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
//CHECK-NEXT:    rng = new mkl::rng::sobol(dpct::get_default_queue(), 1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:    */
//CHECK-NEXT:  }
//CHECK-NEXT:  ~A(){
//CHECK-NEXT:    delete rng;
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  mkl::rng::sobol* rng;
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
//CHECK-NEXT:    rng = new mkl::rng::sobol(q_ct1, 1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:    */
//CHECK-NEXT:    karg1 = sycl::malloc_device<int>(32 , q_ct1);
//CHECK-NEXT:  }
//CHECK-NEXT:  ~B(){
//CHECK-NEXT:    delete rng;
//CHECK-NEXT:    sycl::free(karg1, dpct::get_default_queue());
//CHECK-NEXT:  }
     //CHECK:private:
//CHECK-NEXT:  mkl::rng::sobol* rng;
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
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: Different generator is used, you may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:mkl::rng::philox4x32x10 rng(dpct::get_default_queue(), 1337ull);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: Different generator is used, you may need to adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:mkl::rng::sobol rng(dpct::get_default_queue(), 1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
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
//CHECK:mkl::rng::philox4x32x10 rng(dpct::get_default_queue(), 1337ull);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandCreateGenerator was replaced with 0, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck(0);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was replaced with 0, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck(0);
//CHECK-NEXT:float *d_data;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck([&](){
//CHECK-NEXT:mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:mkl::rng::generate(distr_ct{{[0-9]+}}, rng, 100*100, d_data);
//CHECK-NEXT:return 0;
//CHECK-NEXT:}());
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandDestroyGenerator was replaced with 0, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck(0);
  curandGenerator_t rng;
  curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
  float *d_data;
  curandErrCheck(curandGenerateUniform(rng, d_data, 100*100));
  curandErrCheck(curandDestroyGenerator(rng));
}

void bar4(){
//CHECK:/*
//CHECK-NEXT:DPCT1033:{{[0-9]+}}: Migrated code uses a basic Sobol generator. Initialize mkl::rng::sobol generator with user-defined direction numbers to use it as Scrambled Sobol generator.
//CHECK-NEXT:*/
//CHECK-NEXT:mkl::rng::sobol rng(dpct::get_default_queue(), 1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

//CHECK:int bar5() try {
//CHECK-NEXT:  float *d_data;
//CHECK-NEXT:  mkl::rng::sobol rng2(dpct::get_default_queue(), 1111);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:  */
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:  */
//CHECK-NEXT:  mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:  mkl::rng::generate(distr_ct{{[0-9]+}}, rng2, 100*100, d_data);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
//CHECK-NEXT:  */
//CHECK-NEXT:  return 0;
//CHECK-NEXT:}
int bar5(){
  float *d_data;
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  return curandGenerateUniform(rng2, d_data, 100*100);
}

void bar6(float *x_gpu, size_t n) {
  //CHECK: static mkl::rng::philox4x32x10* gen[16];
  static curandGenerator_t gen[16];
  static int init[16] = {0};
  int i = 0;
  if(!init[i]) {
    //CHECK: gen[i] = new mkl::rng::philox4x32x10(dpct::get_default_queue(), 1234);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
    //CHECK-NEXT: */
    curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen[i], 1234);
    init[i] = 1;
  }
  //CHECK: mkl::rng::uniform<float> distr_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::rng::generate(distr_ct{{[0-9]+}}, *gen[i], n, x_gpu);
  curandGenerateUniform(gen[i], x_gpu, n);
}