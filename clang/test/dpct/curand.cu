// CHECKME
// RUN: cat %s > %T/curand.cu
// RUN: cd %T
//RUN: dpct -out-root %T/curand curand.cu --usm-level=none --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand/curand.dp.cpp --match-full-lines curand.cu
//CHECK:// CHECKME
//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <oneapi/mkl.hpp>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

int main(){
  //CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  //CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed because
  //CHECK-NEXT:this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:float *d_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;

  //CHECK:{
  //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 =
  //CHECK-NEXT:    (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_buf_ct{{[0-9]+}}),
  //CHECK-NEXT:     0);
  //CHECK-NEXT:}
  s1 = curandGenerateUniform(rng, d_data, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 =
  //CHECK-NEXT:    (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_buf_ct{{[0-9]+}}),
  //CHECK-NEXT:     0);
  //CHECK-NEXT:}
  s1 = curandGenerateLogNormal(rng, d_data, 100*100, 123, 456);

  //CHECK:{
  //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 =
  //CHECK-NEXT:    (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_buf_ct{{[0-9]+}}),
  //CHECK-NEXT:     0);
  //CHECK-NEXT:}
  s1 = curandGenerateNormal(rng, d_data, 100*100, 123, 456);

  double* d_data_d;
  //CHECK:{
  //CHECK-NEXT:auto d_data_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(d_data_d);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_d_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerateUniformDouble(rng, d_data_d, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto d_data_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(d_data_d);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_d_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerateLogNormalDouble(rng, d_data_d, 100*100, 123, 456);

  //CHECK:{
  //CHECK-NEXT:auto d_data_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(d_data_d);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_d_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerateNormalDouble(rng, d_data_d, 100*100, 123, 456);

  unsigned int* d_data_ui;

  //CHECK:{
  //CHECK-NEXT:auto d_data_ui_buf_ct{{[0-9]+}} = dpct::get_buffer<uint32_t>(d_data_ui);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100,
  //CHECK-NEXT:                         d_data_ui_buf_ct{{[0-9]+}}),
  //CHECK-NEXT:      0);
  //CHECK-NEXT:}
  s1 = curandGenerate(rng, d_data_ui, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto d_data_ui_buf_ct{{[0-9]+}} = dpct::get_buffer<int32_t>(d_data_ui);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100,
  //CHECK-NEXT:                         d_data_ui_buf_ct{{[0-9]+}}),
  //CHECK-NEXT:      0);
  //CHECK-NEXT:}
  s1 = curandGeneratePoisson(rng, d_data_ui, 100*100, 123.456);

  unsigned long long* d_data_ull;
  //CHECK:{
  //CHECK-NEXT:auto d_data_ull_buf_ct{{[0-9]+}} = dpct::get_buffer<uint64_t>(d_data_ull);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_ull_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerateLongLong(rng, d_data_ull, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto d_data_ull_buf_ct{{[0-9]+}} = dpct::get_buffer<uint64_t>(d_data_ull);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_ull_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  //CHECK-NEXT:codes. 0 is used instead of an error code in an if statement. You may need to
  //CHECK-NEXT:rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if (s1 = 0) {}
  if(s1 = curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:{
  //CHECK-NEXT:  auto d_data_ull_buf_ct{{[0-9]+}} = dpct::get_buffer<uint64_t>(d_data_ull);
  //CHECK-NEXT:  oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_ull_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
  //CHECK-NEXT:codes. 0 is used instead of an error code in an if statement. You may need to
  //CHECK-NEXT:rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if (0) {}
  if(curandGenerateLongLong(rng, d_data_ull, 100*100)){}

  //CHECK:std::shared_ptr<oneapi::mkl::rng::sobol> rng2;
  //CHECK-NEXT:rng2 = std::make_shared<oneapi::mkl::rng::sobol>(q_ct1, 1111);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed
  //CHECK-NEXT: because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
  //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng2, 100 * 100, d_data_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, d_data, 100*100);

  //CHECK:oneapi::mkl::rng::skip_ahead(*rng, 100);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (oneapi::mkl::rng::skip_ahead(*rng2, 200), 0);
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
//CHECK-NEXT:    rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(),
//CHECK-NEXT:                                                    1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed
//CHECK-NEXT:    because this call is redundant in SYCL.
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



void bar1(){
//CHECK:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to
//CHECK-NEXT:adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::mcg59> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::mcg59>(
//CHECK-NEXT:    dpct::get_default_queue(), 1337ull);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed
//CHECK-NEXT:because this call is redundant in SYCL.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:/*
//CHECK-NEXT:DPCT1032:{{[0-9]+}}: A different random number generator is used. You may need to
//CHECK-NEXT:adjust the code.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::sobol> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(),
//CHECK-NEXT:                                                1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed
//CHECK-NEXT:because this call is redundant in SYCL.
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
//CHECK:std::shared_ptr<oneapi::mkl::rng::philox4x32x10> rng;
//CHECK-NEXT:curandErrCheck((rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(
//CHECK-NEXT:                    dpct::get_default_queue(), 1337ull),
//CHECK-NEXT:                0));
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was replaced with
//CHECK-NEXT:0 because this call is redundant in SYCL.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck(0);
//CHECK-NEXT:float *d_data;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the
//CHECK-NEXT:lambda. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:curandErrCheck([&]() {
//CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
//CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100 * 100, d_data_buf_ct{{[0-9]+}});
//CHECK-NEXT:return 0;
//CHECK-NEXT:}());
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
//CHECK:/*
//CHECK-NEXT:DPCT1033:{{[0-9]+}}: Migrated code uses a basic Sobol generator. Initialize
//CHECK-NEXT:oneapi::mkl::rng::sobol generator with user-defined direction numbers to use
//CHECK-NEXT:it as Scrambled Sobol generator.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::rng::sobol> rng;
//CHECK-NEXT:rng = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(),
//CHECK-NEXT:                                                1243);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed
//CHECK-NEXT:because this call is redundant in SYCL.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}

void bar5(){
//CHECK:/*
//CHECK-NEXT:DPCT1036:{{[0-9]+}}: The type curandGenerator_t was not migrated because the migration
//CHECK-NEXT:depends on the second argument of curandCreateGenerator.
//CHECK-NEXT:*/
//CHECK-NEXT:curandGenerator_t rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1028:{{[0-9]+}}: The curandCreateGenerator was not migrated because parameter
//CHECK-NEXT:(curandRngType)101 is unsupported.
//CHECK-NEXT:*/
//CHECK-NEXT:curandCreateGenerator(&rng, (curandRngType)101);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed
//CHECK-NEXT:because this call is redundant in SYCL.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, (curandRngType)101);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}

//CHECK:int bar6() try {
//CHECK-NEXT:  oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:  float *d_data;
//CHECK-NEXT:  std::shared_ptr<oneapi::mkl::rng::sobol> rng2;
//CHECK-NEXT:  rng2 = std::make_shared<oneapi::mkl::rng::sobol>(dpct::get_default_queue(),
//CHECK-NEXT:                                                   1111);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed
//CHECK-NEXT:  because this call is redundant in SYCL.
//CHECK-NEXT:  */
//CHECK-NEXT:  {
//CHECK-NEXT:    auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
//CHECK-NEXT:    oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng2, 100 * 100, d_data_buf_ct{{[0-9]+}});
//CHECK-NEXT:  }
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error
//CHECK-NEXT:  codes. 0 is used instead of an error code in a return statement. You may need
//CHECK-NEXT:  to rewrite this code.
//CHECK-NEXT:  */
//CHECK-NEXT:  return 0;
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
  //CHECK:rng = std::make_shared<oneapi::mkl::rng::philox4x32x10>(
  //CHECK-NEXT:  dpct::cpu_device().default_queue(), 0);
  curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}
