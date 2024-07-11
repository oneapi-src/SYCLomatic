//RUN: dpct -out-root %T/curand-cross-function-placeholder %s --format-range=none --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/curand-cross-function-placeholder/curand-cross-function-placeholder.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/curand-cross-function-placeholder/curand-cross-function-placeholder.dp.cpp -o %T/curand-cross-function-placeholder/curand-cross-function-placeholder.dp.o %}

#include <cuda.h>
#include <stdio.h>
#include <curand.h>

//CHECK: void update(float* randvals, dpct::rng::host_rng_ptr rng, long long nx, long long ny) {
//CHECK-NEXT:   rng->generate_uniform(randvals, nx*ny/2);
//CHECK-NEXT: }
void update(float* randvals, curandGenerator_t rng, long long nx, long long ny) {
  curandGenerateUniform(rng, randvals, nx*ny/2);
}


//CHECK: int main(){
//CHECK-NEXT:   long long nx = 5120;
//CHECK-NEXT:   long long ny = 5120;
//CHECK-NEXT:   unsigned long long seed = 1234ULL;
//CHECK-NEXT:   dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:   dpct::rng::host_rng_ptr rng1;
//CHECK-NEXT:   rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
//CHECK-NEXT:   rng1 = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mrg32k3a);
//CHECK-NEXT:   rng->set_seed(seed);
//CHECK-NEXT:   rng1->set_seed(seed);
//CHECK-NEXT:   float *randvals;
//CHECK-NEXT:   rng->generate_uniform(randvals, nx*ny/2);
//CHECK-NEXT:   update(randvals, rng1, nx, ny);
//CHECK-NEXT:   rng.reset();
//CHECK-NEXT:   rng1.reset();
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
//CHECK-NEXT:   float *randvals;
//CHECK-NEXT:   dpct::rng::host_rng_ptr rng;
//CHECK-NEXT:   rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::philox4x32x10);
//CHECK-NEXT:   rng->set_seed(111);
//CHECK-NEXT:   rng->generate_uniform(randvals, 0);
//CHECK-EMPTY:
//CHECK-NEXT:   rng = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mrg32k3a);
//CHECK-NEXT:   rng->set_seed(222);
//CHECK-NEXT:   rng->generate_uniform(randvals, 0);
//CHECK-NEXT:   rng.reset();
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

