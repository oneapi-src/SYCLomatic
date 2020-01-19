// RUN: cat %s > %T/curand.cu
// RUN: cd %T
//RUN: dpct -out-root %T curand.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand.dp.cpp --match-full-lines curand.cu
//CHECK:#include <CL/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <mkl_rng_sycl.hpp>
#include <cuda.h>
#include <stdio.h>
#include <curand.h>

int main(){
  //CHECK:int s1;
  //CHECK-NEXT:int s2;
  //CHECK-NEXT:mkl::rng::philox4x32x10 rng(dpct::get_default_queue_wait(), 1337ull);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the
  //CHECK-NEXT:function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed,
  //CHECK-NEXT:because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:float *h_data;
  curandStatus_t s1;
  curandStatus s2;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *h_data;

  //CHECK:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data);
  //CHECK-NEXT:cl::sycl::buffer<float> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<float>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(float)));
  //CHECK-NEXT:mkl::rng::uniform<float> distr_ct1;
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerateUniform(rng, h_data, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data);
  //CHECK-NEXT:cl::sycl::buffer<float> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<float>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(float)));
  //CHECK-NEXT:mkl::rng::uniform<float> distr_ct1;
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1), 0);
  //CHECK-NEXT:}
  s1 = curandGenerateUniform(rng, h_data, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data);
  //CHECK-NEXT:cl::sycl::buffer<float> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<float>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(float)));
  //CHECK-NEXT:mkl::rng::lognormal<float> distr_ct1(123, 456, 0.0, 1.0);
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1), 0);
  //CHECK-NEXT:}
  s1 = curandGenerateLogNormal(rng, h_data, 100*100, 123, 456);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data);
  //CHECK-NEXT:cl::sycl::buffer<float> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<float>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(float)));
  //CHECK-NEXT:mkl::rng::gaussian<float> distr_ct1(123, 456);
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1), 0);
  //CHECK-NEXT:}
  s1 = curandGenerateNormal(rng, h_data, 100*100, 123, 456);

  double* h_data_d;
  //CHECK:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_d);
  //CHECK-NEXT:cl::sycl::buffer<double> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<double>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(double)));
  //CHECK-NEXT:mkl::rng::uniform<double> distr_ct1;
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerateUniformDouble(rng, h_data_d, 100*100);

  //CHECK:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_d);
  //CHECK-NEXT:cl::sycl::buffer<double> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<double>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(double)));
  //CHECK-NEXT:mkl::rng::lognormal<double> distr_ct1(123, 456, 0.0, 1.0);
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerateLogNormalDouble(rng, h_data_d, 100*100, 123, 456);

  //CHECK:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_d);
  //CHECK-NEXT:cl::sycl::buffer<double> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<double>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(double)));
  //CHECK-NEXT:mkl::rng::gaussian<double> distr_ct1(123, 456);
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerateNormalDouble(rng, h_data_d, 100*100, 123, 456);

  unsigned int* h_data_ui;
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_ui);
  //CHECK-NEXT:cl::sycl::buffer<uint32_t> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<uint32_t>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(uint32_t)));
  //CHECK-NEXT:mkl::rng::uniform_bits<uint32_t> distr_ct1;
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1), 0);
  //CHECK-NEXT:}
  s1 = curandGenerate(rng, h_data_ui, 100*100);

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_ui);
  //CHECK-NEXT:cl::sycl::buffer<int32_t> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<int32_t>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(int32_t)));
  //CHECK-NEXT:mkl::rng::poisson<int32_t> distr_ct1(123.456);
  //CHECK-NEXT:s1 = (mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1), 0);
  //CHECK-NEXT:}
  s1 = curandGeneratePoisson(rng, h_data_ui, 100*100, 123.456);

  unsigned long long* h_data_ull;
  //CHECK:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data_ull);
  //CHECK-NEXT:cl::sycl::buffer<uint64_t> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<uint64_t>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(uint64_t)));
  //CHECK-NEXT:mkl::rng::uniform_bits<uint64_t> distr_ct1;
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerateLongLong(rng, h_data_ull, 100*100);

  //CHECK:if (s1 = [&]() {
  //CHECK-NEXT:auto allocation_ct1 =
  //CHECK-NEXT:    dpct::mem_mgr::instance().translate_ptr(h_data_ull);
  //CHECK-NEXT:cl::sycl::buffer<uint64_t> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<uint64_t>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(uint64_t)));
  //CHECK-NEXT:mkl::rng::uniform_bits<uint64_t> distr_ct1;
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, 100 * 100, buffer_ct1);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:    }()) {}
  if(s1 = curandGenerateLongLong(rng, h_data_ull, 100*100)){}

  //CHECK:mkl::rng::sobol rng2(dpct::get_default_queue_wait(), 1111);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandCreateGenerator was removed, because the
  //CHECK-NEXT:function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed,
  //CHECK-NEXT:because the function call is redundant in DPC++.
  //CHECK-NEXT:*/
  //CHECK-NEXT:{
  //CHECK-NEXT:auto allocation_ct1 = dpct::mem_mgr::instance().translate_ptr(h_data);
  //CHECK-NEXT:cl::sycl::buffer<float> buffer_ct1 =
  //CHECK-NEXT:    allocation_ct1.buffer.reinterpret<float>(
  //CHECK-NEXT:        cl::sycl::range<1>(allocation_ct1.size / sizeof(float)));
  //CHECK-NEXT:mkl::rng::uniform<float> distr_ct1;
  //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng2, 100 * 100, buffer_ct1);
  //CHECK-NEXT:}
  curandGenerator_t rng2;
  curandCreateGenerator(&rng2, CURAND_RNG_QUASI_DEFAULT);
  curandSetQuasiRandomGeneratorDimensions(rng2, 1111);
  curandGenerateUniform(rng2, h_data, 100*100);

  //CHECK:mkl::rng::skip_ahead(rng, 100);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You
  //CHECK-NEXT:may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:s1 = (mkl::rng::skip_ahead(rng2, 200), 0);
  curandSetGeneratorOffset(rng, 100);
  s1 = curandSetGeneratorOffset(rng2, 200);

  //CHECK:s1 = 0;
  curandDestroyGenerator(rng);
  s1 = curandDestroyGenerator(rng);
}

//CHECK:int foo1();
curandStatus_t foo1();
//CHECK:int foo2();
curandStatus foo2();

//CHECK:class A{
//CHECK-NEXT:public:
//CHECK-NEXT:  void create(){
//CHECK-NEXT:    rng = mkl::rng::sobol(dpct::get_default_queue_wait(), 1243);
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was
//CHECK-NEXT:    removed, because the function call is redundant in DPC++.
//CHECK-NEXT:    */
//CHECK-NEXT:  }

//CHECK:private:
//CHECK-NEXT:  mkl::rng::sobol rng;
//CHECK-NEXT:};
class A{
public:
  void create(){
    curandCreateGenerator(&rng, CURAND_RNG_QUASI_DEFAULT);
    curandSetQuasiRandomGeneratorDimensions(rng, 1243);
  }

private:
  curandGenerator_t rng;
};



void bar1(){
//CHECK:curandGenerator_t rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1028:{{[0-9]+}}: The curandCreateGenerator was not migrated, because parameter
//CHECK-NEXT:CURAND_RNG_PSEUDO_XORWOW is unsupported.
//CHECK-NEXT:*/
//CHECK-NEXT:curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed,
//CHECK-NEXT:because the function call is redundant in DPC++.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
}


void bar2(){
//CHECK:curandGenerator_t rng;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1028:{{[0-9]+}}: The curandCreateGenerator was not migrated, because parameter
//CHECK-NEXT:CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 is unsupported.
//CHECK-NEXT:*/
//CHECK-NEXT:curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to curandSetQuasiRandomGeneratorDimensions was removed,
//CHECK-NEXT:because the function call is redundant in DPC++.
//CHECK-NEXT:*/
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
  curandSetQuasiRandomGeneratorDimensions(rng, 1243);
}
