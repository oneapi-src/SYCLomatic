#include "curand_kernel.h"

__global__ void test(float f1, float f2) {
  // Start
  curandStateScrambledSobol64_t *ps1;
  curandStateSobol64_t *ps2;
  curandStateScrambledSobol32_t *ps3;
  curandStateSobol32_t *ps4;
  curandStateMtgp32_t *ps5;
  curandStateMRG32k3a_t *ps6;
  curandStatePhilox4_32_10_t *ps7;
  curandStateXORWOW_t *ps8;
  curand_log_normal(ps1 /*curandStateScrambledSobol64_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps2 /*curandStateSobol64_t **/, f1 /*float*/, f2 /*float*/);
  curand_log_normal(ps3 /*curandStateScrambledSobol32_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps4 /*curandStateSobol32_t **/, f1 /*float*/, f2 /*float*/);
  curand_log_normal(ps5 /*curandStateMtgp32_t **/, f1 /*float*/, f2 /*float*/);
  curand_log_normal(ps6 /*curandStateMRG32k3a_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps7 /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
                    f2 /*float*/);
  curand_log_normal(ps8 /*curandStateXORWOW_t **/, f1 /*float*/, f2 /*float*/);
  // End
}
