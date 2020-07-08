//RUN: dpct -out-root %T %s --format-range=none --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-cross-function.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <stdio.h>
#include <curand.h>

//CHECK: void update(float* randvals, oneapi::mkl::rng::philox4x32x10* rng, long long nx, long long ny) {
//CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, nx*ny/2, randvals);
//CHECK-NEXT: }
void update(float* randvals, curandGenerator_t rng, long long nx, long long ny) {
  curandGenerateUniform(rng, randvals, nx*ny/2);
}


//CHECK: int main(){
//CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
//CHECK-NEXT:   long long nx = 5120;
//CHECK-NEXT:   long long ny = 5120;
//CHECK-NEXT:   unsigned long long seed = 1234ULL;
//CHECK-NEXT:   oneapi::mkl::rng::philox4x32x10* rng;
//CHECK-NEXT:   rng = new oneapi::mkl::rng::philox4x32x10(dpct::get_default_queue(), seed);
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
//CHECK-NEXT:   */
//CHECK-NEXT:   float *randvals;
//CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, nx*ny/2, randvals);
//CHECK-NEXT:   update(randvals, rng, nx, ny);
//CHECK-NEXT:   delete rng;
//CHECK-NEXT: }
int main(){
  long long nx = 5120;
  long long ny = 5120;
  unsigned long long seed = 1234ULL;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, seed);
  float *randvals;
  curandGenerateUniform(rng, randvals, nx*ny/2);
  update(randvals, rng, nx, ny);
  curandDestroyGenerator(rng);
}