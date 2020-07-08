//RUN: dpct -out-root %T %s --format-range=none --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-cross-function-placeholder.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <stdio.h>
#include <curand.h>

//CHECK: /*
//CHECK-NEXT: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
//CHECK-NEXT: */
//CHECK-NEXT: void update(float* randvals, dpct_placeholder/*Fix the engine type manually*/* rng, long long nx, long long ny) {
//CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, nx*ny/2, randvals);
//CHECK-NEXT: }
void update(float* randvals, curandGenerator_t rng, long long nx, long long ny) {
  curandGenerateUniform(rng, randvals, nx*ny/2);
}


//CHECK: int main(){
//CHECK-NEXT:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:   sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:   long long nx = 5120;
//CHECK-NEXT:   long long ny = 5120;
//CHECK-NEXT:   unsigned long long seed = 1234ULL;
//CHECK-NEXT:   oneapi::mkl::rng::philox4x32x10* rng;
//CHECK-NEXT:   oneapi::mkl::rng::mrg32k3a* rng1;
//CHECK-NEXT:   rng = new oneapi::mkl::rng::philox4x32x10(q_ct1, seed);
//CHECK-NEXT:   rng1 = new oneapi::mkl::rng::mrg32k3a(q_ct1, seed);
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:   */
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:   */
//CHECK-NEXT:   float *randvals;
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, nx*ny/2, randvals);
//CHECK-NEXT:   update(randvals, rng1, nx, ny);
//CHECK-NEXT:   delete rng;
//CHECK-NEXT:   delete rng1;
//CHECK-NEXT: }
int main(){
  long long nx = 5120;
  long long ny = 5120;
  unsigned long long seed = 1234ULL;
  curandGenerator_t rng;
  curandGenerator_t rng1;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandCreateGenerator(&rng1, CURAND_RNG_PSEUDO_MRG32K3A);
  curandSetPseudoRandomGeneratorSeed(rng, seed);
  curandSetPseudoRandomGeneratorSeed(rng1, seed);
  float *randvals;
  curandGenerateUniform(rng, randvals, nx*ny/2);
  update(randvals, rng1, nx, ny);
  curandDestroyGenerator(rng);
  curandDestroyGenerator(rng1);
}

//CHECK: void foo(){
//CHECK-NEXT:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:   sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:   float *randvals;
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   dpct_placeholder/*Fix the engine type manually*/* rng;
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   rng = new dpct_placeholder/*Fix the engine type manually*/(q_ct1, 222);
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:   */
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 0, randvals);
//CHECK-EMPTY:
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
//CHECK-NEXT:   */
//CHECK-NEXT:   rng = new dpct_placeholder/*Fix the engine type manually*/(q_ct1, 222);
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:   */
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 0, randvals);
//CHECK-NEXT:   delete rng;
//CHECK-NEXT: }
void foo(){
  float *randvals;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 111);
  curandGenerateUniform(rng, randvals, 0);

  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MRG32K3A);
  curandSetPseudoRandomGeneratorSeed(rng, 222);
  curandGenerateUniform(rng, randvals, 0);
  curandDestroyGenerator(rng);
}