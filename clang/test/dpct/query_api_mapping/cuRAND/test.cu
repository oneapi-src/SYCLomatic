// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Host API

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandCreateGenerator | FileCheck %s -check-prefix=CURANDCREATEGENERATOR
// CURANDCREATEGENERATOR: CUDA API:
// CURANDCREATEGENERATOR-NEXT:   curandCreateGenerator(pg /*curandGenerator_t **/, r /*curandRngType_t*/);
// CURANDCREATEGENERATOR-NEXT: Is migrated to:
// CURANDCREATEGENERATOR-NEXT:   *(pg) = dpct::rng::create_host_rng(r);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandCreateGeneratorHost | FileCheck %s -check-prefix=CURANDCREATEGENERATORHOST
// CURANDCREATEGENERATORHOST: CUDA API:
// CURANDCREATEGENERATORHOST-NEXT:   curandCreateGeneratorHost(pg /*curandGenerator_t **/, r /*curandRngType_t*/);
// CURANDCREATEGENERATORHOST-NEXT: Is migrated to:
// CURANDCREATEGENERATORHOST-NEXT:   *(pg) = dpct::rng::create_host_rng(r, dpct::cpu_device().default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandDestroyGenerator | FileCheck %s -check-prefix=CURANDDESTROYGENERATOR
// CURANDDESTROYGENERATOR: CUDA API:
// CURANDDESTROYGENERATOR-NEXT:   curandGenerator_t g;
// CURANDDESTROYGENERATOR-NEXT:   curandDestroyGenerator(g /*curandGenerator_t*/);
// CURANDDESTROYGENERATOR-NEXT: Is migrated to:
// CURANDDESTROYGENERATOR-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDDESTROYGENERATOR-NEXT:   g.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerate | FileCheck %s -check-prefix=CURANDGENERATE
// CURANDGENERATE: CUDA API:
// CURANDGENERATE-NEXT:   curandGenerator_t g;
// CURANDGENERATE-NEXT:   curandGenerate(g /*curandGenerator_t*/, pu /*unsigned int **/, s /*size_t*/);
// CURANDGENERATE-NEXT: Is migrated to:
// CURANDGENERATE-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATE-NEXT:   g->generate_uniform_bits(pu, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateLogNormal | FileCheck %s -check-prefix=CURANDGENERATELOGNORMAL
// CURANDGENERATELOGNORMAL: CUDA API:
// CURANDGENERATELOGNORMAL-NEXT:   curandGenerator_t g;
// CURANDGENERATELOGNORMAL-NEXT:   curandGenerateLogNormal(g /*curandGenerator_t*/, pf /*float **/, s /*size_t*/,
// CURANDGENERATELOGNORMAL-NEXT:                           f1 /*float*/, f2 /*float*/);
// CURANDGENERATELOGNORMAL-NEXT: Is migrated to:
// CURANDGENERATELOGNORMAL-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATELOGNORMAL-NEXT:   g->generate_lognormal(pf, s, f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateLogNormalDouble | FileCheck %s -check-prefix=CURANDGENERATELOGNORMALDOUBLE
// CURANDGENERATELOGNORMALDOUBLE: CUDA API:
// CURANDGENERATELOGNORMALDOUBLE-NEXT:   curandGenerator_t g;
// CURANDGENERATELOGNORMALDOUBLE-NEXT:   curandGenerateLogNormalDouble(g /*curandGenerator_t*/, pd /*double **/,
// CURANDGENERATELOGNORMALDOUBLE-NEXT:                                 s /*size_t*/, d1 /*double*/, d2 /*double*/);
// CURANDGENERATELOGNORMALDOUBLE-NEXT: Is migrated to:
// CURANDGENERATELOGNORMALDOUBLE-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATELOGNORMALDOUBLE-NEXT:   g->generate_lognormal(pd, s, d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateLongLong | FileCheck %s -check-prefix=CURANDGENERATELONGLONG
// CURANDGENERATELONGLONG: CUDA API:
// CURANDGENERATELONGLONG-NEXT:   curandGenerator_t g;
// CURANDGENERATELONGLONG-NEXT:   curandGenerateLongLong(g /*curandGenerator_t*/, ull /*unsigned long long **/,
// CURANDGENERATELONGLONG-NEXT:                          s /*size_t*/);
// CURANDGENERATELONGLONG-NEXT: Is migrated to:
// CURANDGENERATELONGLONG-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATELONGLONG-NEXT:   g->generate_uniform_bits(ull, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateNormal | FileCheck %s -check-prefix=CURANDGENERATENORMAL
// CURANDGENERATENORMAL: CUDA API:
// CURANDGENERATENORMAL-NEXT:   curandGenerator_t g;
// CURANDGENERATENORMAL-NEXT:   curandGenerateNormal(g /*curandGenerator_t*/, pf /*float **/, s /*size_t*/,
// CURANDGENERATENORMAL-NEXT:                        f1 /*float*/, f2 /*float*/);
// CURANDGENERATENORMAL-NEXT: Is migrated to:
// CURANDGENERATENORMAL-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATENORMAL-NEXT:   g->generate_gaussian(pf, s, f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateNormalDouble | FileCheck %s -check-prefix=CURANDGENERATENORMALDOUBLE
// CURANDGENERATENORMALDOUBLE: CUDA API:
// CURANDGENERATENORMALDOUBLE-NEXT:   curandGenerator_t g;
// CURANDGENERATENORMALDOUBLE-NEXT:   curandGenerateNormalDouble(g /*curandGenerator_t*/, pd /*double **/,
// CURANDGENERATENORMALDOUBLE-NEXT:                              s /*size_t*/, d1 /*double*/, d2 /*double*/);
// CURANDGENERATENORMALDOUBLE-NEXT: Is migrated to:
// CURANDGENERATENORMALDOUBLE-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATENORMALDOUBLE-NEXT:   g->generate_gaussian(pd, s, d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGeneratePoisson | FileCheck %s -check-prefix=CURANDGENERATEPOISSON
// CURANDGENERATEPOISSON: CUDA API:
// CURANDGENERATEPOISSON-NEXT:   curandGenerator_t g;
// CURANDGENERATEPOISSON-NEXT:   curandGeneratePoisson(g /*curandGenerator_t*/, pu /*unsigned int **/,
// CURANDGENERATEPOISSON-NEXT:                         s /*size_t*/, d /*double*/);
// CURANDGENERATEPOISSON-NEXT: Is migrated to:
// CURANDGENERATEPOISSON-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATEPOISSON-NEXT:   g->generate_poisson(pu, s, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateUniform | FileCheck %s -check-prefix=CURANDGENERATEUNIFORM
// CURANDGENERATEUNIFORM: CUDA API:
// CURANDGENERATEUNIFORM-NEXT:   curandGenerator_t g;
// CURANDGENERATEUNIFORM-NEXT:   curandGenerateUniform(g /*curandGenerator_t*/, pf /*float **/, s /*size_t*/);
// CURANDGENERATEUNIFORM-NEXT: Is migrated to:
// CURANDGENERATEUNIFORM-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATEUNIFORM-NEXT:   g->generate_uniform(pf, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandGenerateUniformDouble | FileCheck %s -check-prefix=CURANDGENERATEUNIFORMDOUBLE
// CURANDGENERATEUNIFORMDOUBLE: CUDA API:
// CURANDGENERATEUNIFORMDOUBLE-NEXT:   curandGenerator_t g;
// CURANDGENERATEUNIFORMDOUBLE-NEXT:   curandGenerateUniformDouble(g /*curandGenerator_t*/, pd /*double **/,
// CURANDGENERATEUNIFORMDOUBLE-NEXT:                               s /*size_t*/);
// CURANDGENERATEUNIFORMDOUBLE-NEXT: Is migrated to:
// CURANDGENERATEUNIFORMDOUBLE-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDGENERATEUNIFORMDOUBLE-NEXT:   g->generate_uniform(pd, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandSetGeneratorOffset | FileCheck %s -check-prefix=CURANDSETGENERATOROFFSET
// CURANDSETGENERATOROFFSET: CUDA API:
// CURANDSETGENERATOROFFSET-NEXT:   curandGenerator_t g;
// CURANDSETGENERATOROFFSET-NEXT:   curandSetGeneratorOffset(g /*curandGenerator_t*/, ull /*unsigned long long*/);
// CURANDSETGENERATOROFFSET-NEXT: Is migrated to:
// CURANDSETGENERATOROFFSET-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDSETGENERATOROFFSET-NEXT:   g->skip_ahead(ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandSetGeneratorOrdering | FileCheck %s -check-prefix=CURANDSETGENERATORORDERING
// CURANDSETGENERATORORDERING: CUDA API:
// CURANDSETGENERATORORDERING-NEXT:   curandGenerator_t g;
// CURANDSETGENERATORORDERING-NEXT:   curandSetGeneratorOrdering(g /*curandGenerator_t*/, o /*curandOrdering_t*/);
// CURANDSETGENERATORORDERING-NEXT: Is migrated to:
// CURANDSETGENERATORORDERING-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDSETGENERATORORDERING-NEXT:   g->set_mode(o);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandSetPseudoRandomGeneratorSeed | FileCheck %s -check-prefix=CURANDSETPSEUDORANDOMGENERATORSEED
// CURANDSETPSEUDORANDOMGENERATORSEED: CUDA API:
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT:   curandGenerator_t g;
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT:   curandSetPseudoRandomGeneratorSeed(g /*curandGenerator_t*/,
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT:                                      ull /*unsigned long long*/);
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT: Is migrated to:
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDSETPSEUDORANDOMGENERATORSEED-NEXT:   g->set_seed(ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandSetQuasiRandomGeneratorDimensions | FileCheck %s -check-prefix=CURANDSETQUASIRANDOMGENERATORDIMENSIONS
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS: CUDA API:
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT:   curandGenerator_t g;
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT:   curandSetQuasiRandomGeneratorDimensions(g /*curandGenerator_t*/,
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT:                                           u /*unsigned int*/);
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT: Is migrated to:
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDSETQUASIRANDOMGENERATORDIMENSIONS-NEXT:   g->set_dimensions(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curandSetStream | FileCheck %s -check-prefix=CURANDSETSTREAM
// CURANDSETSTREAM: CUDA API:
// CURANDSETSTREAM-NEXT:   curandGenerator_t g;
// CURANDSETSTREAM-NEXT:   curandSetStream(g /*curandGenerator_t*/, s /*cudaStream_t*/);
// CURANDSETSTREAM-NEXT: Is migrated to:
// CURANDSETSTREAM-NEXT:   dpct::rng::host_rng_ptr g;
// CURANDSETSTREAM-NEXT:   g->set_queue(s);

/// Device API

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand | FileCheck %s -check-prefix=CURAND
// CURAND: CUDA API:
// CURAND-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND-NEXT:   curandStateXORWOW_t *ps3;
// CURAND-NEXT:   curand(ps1 /*curandStateMRG32k3a_t **/);
// CURAND-NEXT:   curand(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND-NEXT:   curand(ps3 /*curandStateXORWOW_t **/);
// CURAND-NEXT: Is migrated to:
// CURAND-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND-NEXT:   ps1->generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
// CURAND-NEXT:   ps2->generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
// CURAND-NEXT:   ps3->generate<oneapi::mkl::rng::device::uniform_bits<std::uint32_t>, 1>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand4 | FileCheck %s -check-prefix=CURAND4
// CURAND4: CUDA API:
// CURAND4-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND4-NEXT:   curand4(ps /*curandStatePhilox4_32_10_t **/);
// CURAND4-NEXT: Is migrated to:
// CURAND4-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND4-NEXT:   ps->generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 4>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_init | FileCheck %s -check-prefix=CURAND_INIT
// CURAND_INIT: CUDA API:
// CURAND_INIT-NEXT:   curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
// CURAND_INIT-NEXT:               ull3 /*unsigned long long*/, ps1 /*curandStateMRG32k3a_t **/);
// CURAND_INIT-NEXT:   curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
// CURAND_INIT-NEXT:               ull3 /*unsigned long long*/,
// CURAND_INIT-NEXT:               ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_INIT-NEXT:   curand_init(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/,
// CURAND_INIT-NEXT:               ull3 /*unsigned long long*/, ps3 /*curandStateXORWOW_t **/);
// CURAND_INIT-NEXT: Is migrated to:
// CURAND_INIT-NEXT:   *ps1 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>>(ull1, {static_cast<std::uint64_t>(ull3), static_cast<std::uint64_t>(ull2 * 8)});
// CURAND_INIT-NEXT:   *ps2 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(ull1, {static_cast<std::uint64_t>(ull3), static_cast<std::uint64_t>(ull2 * 4)});
// CURAND_INIT-NEXT:   *ps3 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(ull1, static_cast<std::uint64_t>(ull3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_log_normal | FileCheck %s -check-prefix=CURAND_LOG_NORMAL
// CURAND_LOG_NORMAL: CUDA API:
// CURAND_LOG_NORMAL-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_LOG_NORMAL-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_LOG_NORMAL-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_LOG_NORMAL-NEXT:   curand_log_normal(ps1 /*curandStateMRG32k3a_t **/, f1 /*float*/,
// CURAND_LOG_NORMAL-NEXT:                     f2 /*float*/);
// CURAND_LOG_NORMAL-NEXT:   curand_log_normal(ps2 /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
// CURAND_LOG_NORMAL-NEXT:                     f2 /*float*/);
// CURAND_LOG_NORMAL-NEXT:   curand_log_normal(ps3 /*curandStateXORWOW_t **/, f1 /*float*/, f2 /*float*/);
// CURAND_LOG_NORMAL-NEXT: Is migrated to:
// CURAND_LOG_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_LOG_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_LOG_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_LOG_NORMAL-NEXT:   ps1->generate<oneapi::mkl::rng::device::lognormal<float>, 1>(f1, f2);
// CURAND_LOG_NORMAL-NEXT:   ps2->generate<oneapi::mkl::rng::device::lognormal<float>, 1>(f1, f2);
// CURAND_LOG_NORMAL-NEXT:   ps3->generate<oneapi::mkl::rng::device::lognormal<float>, 1>(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_log_normal2 | FileCheck %s -check-prefix=CURAND_LOG_NORMAL2
// CURAND_LOG_NORMAL2: CUDA API:
// CURAND_LOG_NORMAL2-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_LOG_NORMAL2-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_LOG_NORMAL2-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_LOG_NORMAL2-NEXT:   curand_log_normal2(ps1 /*curandStateMRG32k3a_t **/, f1 /*float*/,
// CURAND_LOG_NORMAL2-NEXT:                      f2 /*float*/);
// CURAND_LOG_NORMAL2-NEXT:   curand_log_normal2(ps2 /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
// CURAND_LOG_NORMAL2-NEXT:                      f2 /*float*/);
// CURAND_LOG_NORMAL2-NEXT:   curand_log_normal2(ps3 /*curandStateXORWOW_t **/, f1 /*float*/, f2 /*float*/);
// CURAND_LOG_NORMAL2-NEXT: Is migrated to:
// CURAND_LOG_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_LOG_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_LOG_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_LOG_NORMAL2-NEXT:   ps1->generate<oneapi::mkl::rng::device::lognormal<float>, 2>(f1, f2);
// CURAND_LOG_NORMAL2-NEXT:   ps2->generate<oneapi::mkl::rng::device::lognormal<float>, 2>(f1, f2);
// CURAND_LOG_NORMAL2-NEXT:   ps3->generate<oneapi::mkl::rng::device::lognormal<float>, 2>(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_log_normal2_double | FileCheck %s -check-prefix=CURAND_LOG_NORMAL2_DOUBLE
// CURAND_LOG_NORMAL2_DOUBLE: CUDA API:
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curand_log_normal2_double(ps1 /*curandStateMRG32k3a_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:                             d2 /*double*/);
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curand_log_normal2_double(ps2 /*curandStatePhilox4_32_10_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:                             d2 /*double*/);
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   curand_log_normal2_double(ps3 /*curandStateXORWOW_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:                             d2 /*double*/);
// CURAND_LOG_NORMAL2_DOUBLE-NEXT: Is migrated to:
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   ps1->generate<oneapi::mkl::rng::device::lognormal<double>, 2>(d1, d2);
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   ps2->generate<oneapi::mkl::rng::device::lognormal<double>, 2>(d1, d2);
// CURAND_LOG_NORMAL2_DOUBLE-NEXT:   ps3->generate<oneapi::mkl::rng::device::lognormal<double>, 2>(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_log_normal4 | FileCheck %s -check-prefix=CURAND_LOG_NORMAL4
// CURAND_LOG_NORMAL4: CUDA API:
// CURAND_LOG_NORMAL4-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND_LOG_NORMAL4-NEXT:   curand_log_normal4(ps /*curandStatePhilox4_32_10_t **/, f1 /*float*/,
// CURAND_LOG_NORMAL4-NEXT:                      f2 /*float*/);
// CURAND_LOG_NORMAL4-NEXT: Is migrated to:
// CURAND_LOG_NORMAL4-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND_LOG_NORMAL4-NEXT:   ps->generate<oneapi::mkl::rng::device::lognormal<float>, 4>(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_log_normal_double | FileCheck %s -check-prefix=CURAND_LOG_NORMAL_DOUBLE
// CURAND_LOG_NORMAL_DOUBLE: CUDA API:
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curand_log_normal_double(ps1 /*curandStateMRG32k3a_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL_DOUBLE-NEXT:                            d2 /*double*/);
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curand_log_normal_double(ps2 /*curandStatePhilox4_32_10_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL_DOUBLE-NEXT:                            d2 /*double*/);
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   curand_log_normal_double(ps3 /*curandStateXORWOW_t **/, d1 /*double*/,
// CURAND_LOG_NORMAL_DOUBLE-NEXT:                            d2 /*double*/);
// CURAND_LOG_NORMAL_DOUBLE-NEXT: Is migrated to:
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   ps1->generate<oneapi::mkl::rng::device::lognormal<double>, 1>(d1, d2);
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   ps2->generate<oneapi::mkl::rng::device::lognormal<double>, 1>(d1, d2);
// CURAND_LOG_NORMAL_DOUBLE-NEXT:   ps3->generate<oneapi::mkl::rng::device::lognormal<double>, 1>(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_normal | FileCheck %s -check-prefix=CURAND_NORMAL
// CURAND_NORMAL: CUDA API:
// CURAND_NORMAL-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_NORMAL-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_NORMAL-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_NORMAL-NEXT:   curand_normal(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_NORMAL-NEXT:   curand_normal(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_NORMAL-NEXT:   curand_normal(ps3 /*curandStateXORWOW_t **/);
// CURAND_NORMAL-NEXT: Is migrated to:
// CURAND_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_NORMAL-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_NORMAL-NEXT:   ps1->generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
// CURAND_NORMAL-NEXT:   ps2->generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
// CURAND_NORMAL-NEXT:   ps3->generate<oneapi::mkl::rng::device::gaussian<float>, 1>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_normal2 | FileCheck %s -check-prefix=CURAND_NORMAL2
// CURAND_NORMAL2: CUDA API:
// CURAND_NORMAL2-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_NORMAL2-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_NORMAL2-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_NORMAL2-NEXT:   curand_normal2(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_NORMAL2-NEXT:   curand_normal2(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_NORMAL2-NEXT:   curand_normal2(ps3 /*curandStateXORWOW_t **/);
// CURAND_NORMAL2-NEXT: Is migrated to:
// CURAND_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_NORMAL2-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_NORMAL2-NEXT:   ps1->generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
// CURAND_NORMAL2-NEXT:   ps2->generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
// CURAND_NORMAL2-NEXT:   ps3->generate<oneapi::mkl::rng::device::gaussian<float>, 2>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_normal2_double | FileCheck %s -check-prefix=CURAND_NORMAL2_DOUBLE
// CURAND_NORMAL2_DOUBLE: CUDA API:
// CURAND_NORMAL2_DOUBLE-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_NORMAL2_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_NORMAL2_DOUBLE-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_NORMAL2_DOUBLE-NEXT:   curand_normal2_double(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_NORMAL2_DOUBLE-NEXT:   curand_normal2_double(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_NORMAL2_DOUBLE-NEXT:   curand_normal2_double(ps3 /*curandStateXORWOW_t **/);
// CURAND_NORMAL2_DOUBLE-NEXT: Is migrated to:
// CURAND_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_NORMAL2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_NORMAL2_DOUBLE-NEXT:   ps1->generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
// CURAND_NORMAL2_DOUBLE-NEXT:   ps2->generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
// CURAND_NORMAL2_DOUBLE-NEXT:   ps3->generate<oneapi::mkl::rng::device::gaussian<double>, 2>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_normal4 | FileCheck %s -check-prefix=CURAND_NORMAL4
// CURAND_NORMAL4: CUDA API:
// CURAND_NORMAL4-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND_NORMAL4-NEXT:   curand_normal4(ps /*curandStatePhilox4_32_10_t **/);
// CURAND_NORMAL4-NEXT: Is migrated to:
// CURAND_NORMAL4-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND_NORMAL4-NEXT:   ps->generate<oneapi::mkl::rng::device::gaussian<float>, 4>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_normal_double | FileCheck %s -check-prefix=CURAND_NORMAL_DOUBLE
// CURAND_NORMAL_DOUBLE: CUDA API:
// CURAND_NORMAL_DOUBLE-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_NORMAL_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_NORMAL_DOUBLE-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_NORMAL_DOUBLE-NEXT:   curand_normal_double(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_NORMAL_DOUBLE-NEXT:   curand_normal_double(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_NORMAL_DOUBLE-NEXT:   curand_normal_double(ps3 /*curandStateXORWOW_t **/);
// CURAND_NORMAL_DOUBLE-NEXT: Is migrated to:
// CURAND_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_NORMAL_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_NORMAL_DOUBLE-NEXT:   ps1->generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
// CURAND_NORMAL_DOUBLE-NEXT:   ps2->generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
// CURAND_NORMAL_DOUBLE-NEXT:   ps3->generate<oneapi::mkl::rng::device::gaussian<double>, 1>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_poisson | FileCheck %s -check-prefix=CURAND_POISSON
// CURAND_POISSON: CUDA API:
// CURAND_POISSON-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_POISSON-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_POISSON-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_POISSON-NEXT:   curand_poisson(ps1 /*curandStateMRG32k3a_t **/, d /*double*/);
// CURAND_POISSON-NEXT:   curand_poisson(ps2 /*curandStatePhilox4_32_10_t **/, d /*double*/);
// CURAND_POISSON-NEXT:   curand_poisson(ps3 /*curandStateXORWOW_t **/, d /*double*/);
// CURAND_POISSON-NEXT: Is migrated to:
// CURAND_POISSON-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_POISSON-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_POISSON-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_POISSON-NEXT:   ps1->generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(d);
// CURAND_POISSON-NEXT:   ps2->generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(d);
// CURAND_POISSON-NEXT:   ps3->generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_poisson4 | FileCheck %s -check-prefix=CURAND_POISSON4
// CURAND_POISSON4: CUDA API:
// CURAND_POISSON4-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND_POISSON4-NEXT:   curand_poisson4(ps /*curandStatePhilox4_32_10_t **/, d /*double*/);
// CURAND_POISSON4-NEXT: Is migrated to:
// CURAND_POISSON4-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND_POISSON4-NEXT:   ps->generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 4>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_uniform | FileCheck %s -check-prefix=CURAND_UNIFORM
// CURAND_UNIFORM: CUDA API:
// CURAND_UNIFORM-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_UNIFORM-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_UNIFORM-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_UNIFORM-NEXT:   curand_uniform(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_UNIFORM-NEXT:   curand_uniform(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_UNIFORM-NEXT:   curand_uniform(ps3 /*curandStateXORWOW_t **/);
// CURAND_UNIFORM-NEXT: Is migrated to:
// CURAND_UNIFORM-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_UNIFORM-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_UNIFORM-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_UNIFORM-NEXT:   ps1->generate<oneapi::mkl::rng::device::uniform<float>, 1>();
// CURAND_UNIFORM-NEXT:   ps2->generate<oneapi::mkl::rng::device::uniform<float>, 1>();
// CURAND_UNIFORM-NEXT:   ps3->generate<oneapi::mkl::rng::device::uniform<float>, 1>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_uniform2_double | FileCheck %s -check-prefix=CURAND_UNIFORM2_DOUBLE
// CURAND_UNIFORM2_DOUBLE: CUDA API:
// CURAND_UNIFORM2_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND_UNIFORM2_DOUBLE-NEXT:   curand_uniform2_double(ps /*curandStatePhilox4_32_10_t **/);
// CURAND_UNIFORM2_DOUBLE-NEXT: Is migrated to:
// CURAND_UNIFORM2_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND_UNIFORM2_DOUBLE-NEXT:   ps->generate<oneapi::mkl::rng::device::uniform<double>, 2>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_uniform4 | FileCheck %s -check-prefix=CURAND_UNIFORM4
// CURAND_UNIFORM4: CUDA API:
// CURAND_UNIFORM4-NEXT:   curandStatePhilox4_32_10_t *ps;
// CURAND_UNIFORM4-NEXT:   curand_uniform4(ps /*curandStatePhilox4_32_10_t **/);
// CURAND_UNIFORM4-NEXT: Is migrated to:
// CURAND_UNIFORM4-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps;
// CURAND_UNIFORM4-NEXT:   ps->generate<oneapi::mkl::rng::device::uniform<float>, 4>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=curand_uniform_double | FileCheck %s -check-prefix=CURAND_UNIFORM_DOUBLE
// CURAND_UNIFORM_DOUBLE: CUDA API:
// CURAND_UNIFORM_DOUBLE-NEXT:   curandStateMRG32k3a_t *ps1;
// CURAND_UNIFORM_DOUBLE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// CURAND_UNIFORM_DOUBLE-NEXT:   curandStateXORWOW_t *ps3;
// CURAND_UNIFORM_DOUBLE-NEXT:   curand_uniform_double(ps1 /*curandStateMRG32k3a_t **/);
// CURAND_UNIFORM_DOUBLE-NEXT:   curand_uniform_double(ps2 /*curandStatePhilox4_32_10_t **/);
// CURAND_UNIFORM_DOUBLE-NEXT:   curand_uniform_double(ps3 /*curandStateXORWOW_t **/);
// CURAND_UNIFORM_DOUBLE-NEXT: Is migrated to:
// CURAND_UNIFORM_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// CURAND_UNIFORM_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// CURAND_UNIFORM_DOUBLE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// CURAND_UNIFORM_DOUBLE-NEXT:   ps1->generate<oneapi::mkl::rng::device::uniform<double>, 1>();
// CURAND_UNIFORM_DOUBLE-NEXT:   ps2->generate<oneapi::mkl::rng::device::uniform<double>, 1>();
// CURAND_UNIFORM_DOUBLE-NEXT:   ps3->generate<oneapi::mkl::rng::device::uniform<double>, 1>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=skipahead_sequence | FileCheck %s -check-prefix=SKIPAHEAD_SEQUENCE
// SKIPAHEAD_SEQUENCE: CUDA API:
// SKIPAHEAD_SEQUENCE-NEXT:   curandStateMRG32k3a_t *ps1;
// SKIPAHEAD_SEQUENCE-NEXT:   curandStatePhilox4_32_10_t *ps2;
// SKIPAHEAD_SEQUENCE-NEXT:   curandStateXORWOW_t *ps3;
// SKIPAHEAD_SEQUENCE-NEXT:   skipahead_sequence(ull, ps1 /*curandStateMRG32k3a_t **/);
// SKIPAHEAD_SEQUENCE-NEXT:   skipahead_sequence(ull, ps2 /*curandStatePhilox4_32_10_t **/);
// SKIPAHEAD_SEQUENCE-NEXT:   skipahead_sequence(ull, ps3 /*curandStateXORWOW_t **/);
// SKIPAHEAD_SEQUENCE-NEXT: Is migrated to:
// SKIPAHEAD_SEQUENCE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps1;
// SKIPAHEAD_SEQUENCE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> *ps2;
// SKIPAHEAD_SEQUENCE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *ps3;
// SKIPAHEAD_SEQUENCE-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps1->get_engine(), {0, ull * (std::uint64_t(1) << 63)});
// SKIPAHEAD_SEQUENCE-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps2->get_engine(), {0, static_cast<std::uint64_t>(ull * 4)});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=skipahead_subsequence | FileCheck %s -check-prefix=SKIPAHEAD_SUBSEQUENCE
// SKIPAHEAD_SUBSEQUENCE: CUDA API:
// SKIPAHEAD_SUBSEQUENCE-NEXT:   curandStateMRG32k3a_t *ps;
// SKIPAHEAD_SUBSEQUENCE-NEXT:   skipahead_subsequence(ull, ps /*curandStateMRG32k3a_t **/);
// SKIPAHEAD_SUBSEQUENCE-NEXT: Is migrated to:
// SKIPAHEAD_SUBSEQUENCE-NEXT:   dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> *ps;
// SKIPAHEAD_SUBSEQUENCE-NEXT:   oneapi::mkl::rng::device::skip_ahead(ps->get_engine(), {0, static_cast<std::uint64_t>(ull * 8)});
