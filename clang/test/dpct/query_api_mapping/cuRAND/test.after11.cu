// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=skipahead | FileCheck %s -check-prefix=SKIPAHEAD
// SKIPAHEAD: CUDA API:
// SKIPAHEAD-NEXT:   curandStateMRG32k3a_t *ps1;
// SKIPAHEAD-NEXT:   curandStatePhilox4_32_10_t *ps2;
// SKIPAHEAD-NEXT:   curandStateXORWOW_t *ps3;
// SKIPAHEAD-NEXT:   skipahead(ull, ps1 /*curandStateMRG32k3a_t **/);
// SKIPAHEAD-NEXT:   skipahead(ull, ps2 /*curandStatePhilox4_32_10_t **/);
// SKIPAHEAD-NEXT:   skipahead(ull, ps3 /*curandStateXORWOW_t **/);
// SKIPAHEAD-NEXT:   skipahead(u, ps1 /*curandStateMRG32k3a_t **/);
// SKIPAHEAD-NEXT:   skipahead(u, ps2 /*curandStatePhilox4_32_10_t **/);
// SKIPAHEAD-NEXT:   skipahead(u, ps3 /*curandStateXORWOW_t **/);
// SKIPAHEAD-NEXT: Is migrated to:
// SKIPAHEAD-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// SKIPAHEAD-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// SKIPAHEAD-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps1->get_engine(), ull);
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps2->get_engine(), ull);
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps3->get_engine(), ull);
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps1->get_engine(), u);
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps2->get_engine(), u);
// SKIPAHEAD-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps3->get_engine(), u);
// SKIPAHEAD-EMPTY:
