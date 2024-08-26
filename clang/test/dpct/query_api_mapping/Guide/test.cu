/// Memory Fence Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__threadfence_block | FileCheck %s -check-prefix=__THREADFENCE_BLOCK
// __THREADFENCE_BLOCK: CUDA API:
// __THREADFENCE_BLOCK-NEXT:   __threadfence_block();
// __THREADFENCE_BLOCK-NEXT: Is migrated to:
// __THREADFENCE_BLOCK-NEXT:   sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__threadfence | FileCheck %s -check-prefix=__THREADFENCE
// __THREADFENCE: CUDA API:
// __THREADFENCE-NEXT:   __threadfence();
// __THREADFENCE-NEXT: Is migrated to:
// __THREADFENCE-NEXT:   sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__threadfence_system | FileCheck %s -check-prefix=__THREADFENCE_SYSTEM
// __THREADFENCE_SYSTEM: CUDA API:
// __THREADFENCE_SYSTEM-NEXT:   __threadfence_system();
// __THREADFENCE_SYSTEM-NEXT: Is migrated to:
// __THREADFENCE_SYSTEM-NEXT:   sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);

/// Synchronization Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__syncthreads | FileCheck %s -check-prefix=__SYNCTHREADS
// __SYNCTHREADS: CUDA API:
// __SYNCTHREADS-NEXT:   __syncthreads();
// __SYNCTHREADS-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// __SYNCTHREADS-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier(sycl::access::fence_space::local_space);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__syncthreads_count | FileCheck %s -check-prefix=__SYNCTHREADSCOUNT
// __SYNCTHREADSCOUNT: CUDA API:
// __SYNCTHREADSCOUNT-NEXT:   __syncthreads_count(i /*int*/);
// __SYNCTHREADSCOUNT-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// __SYNCTHREADSCOUNT-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
// __SYNCTHREADSCOUNT-NEXT:   sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), i == 0 ? 0 : 1, sycl::ext::oneapi::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__syncthreads_and | FileCheck %s -check-prefix=__SYNCTHREADS_AND
// __SYNCTHREADS_AND: CUDA API:
// __SYNCTHREADS_AND-NEXT:   __syncthreads_and(i /*int*/);
// __SYNCTHREADS_AND-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// __SYNCTHREADS_AND-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
// __SYNCTHREADS_AND-NEXT:   sycl::all_of_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__syncthreads_or | FileCheck %s -check-prefix=__SYNCTHREADS_OR
// __SYNCTHREADS_OR: CUDA API:
// __SYNCTHREADS_OR-NEXT:   __syncthreads_or(i /*int*/);
// __SYNCTHREADS_OR-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// __SYNCTHREADS_OR-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
// __SYNCTHREADS_OR-NEXT:   sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), i);

/// Texture Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex1Dfetch | FileCheck %s -check-prefix=TEX1DFETCH
// TEX1DFETCH: CUDA API:
// TEX1DFETCH-NEXT:   tex1Dfetch<T>(t /*cudaTextureObject_t*/, i /*int*/);
// TEX1DFETCH-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX1DFETCH-NEXT:   sycl::ext::oneapi::experimental::sample_image<T>(t, float(i));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex1D | FileCheck %s -check-prefix=TEX1D
// TEX1D: CUDA API:
// TEX1D-NEXT:   tex1D<T>(t /*cudaTextureObject_t*/, f /*float*/);
// TEX1D-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX1D-NEXT:   sycl::ext::oneapi::experimental::sample_image<T>(t, float(f));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex1DLod | FileCheck %s -check-prefix=TEX1DLOD
// TEX1DLOD: CUDA API:
// TEX1DLOD-NEXT:   tex1DLod<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/);
// TEX1DLOD-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX1DLOD-NEXT:   sycl::ext::oneapi::experimental::sample_mipmap<T>(t, float(f1), f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex2D | FileCheck %s -check-prefix=TEX2D
// TEX2D: CUDA API:
// TEX2D-NEXT:   tex2D<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/);
// TEX2D-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX2D-NEXT:   sycl::ext::oneapi::experimental::sample_image<T>(t, sycl::float2(f1, f2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex2DLod | FileCheck %s -check-prefix=TEX2DLOD
// TEX2DLOD: CUDA API:
// TEX2DLOD-NEXT:   tex2DLod<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/,
// TEX2DLOD-NEXT:               f3 /*float*/);
// TEX2DLOD-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX2DLOD-NEXT:   sycl::ext::oneapi::experimental::sample_mipmap<T>(t, sycl::float2(f1, f2), f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex3D | FileCheck %s -check-prefix=TEX3D
// TEX3D: CUDA API:
// TEX3D-NEXT:   tex3D<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/, f3 /*float*/);
// TEX3D-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX3D-NEXT:   sycl::ext::oneapi::experimental::sample_image<T>(t, sycl::float3(f1, f2, f3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tex3DLod | FileCheck %s -check-prefix=TEX3DLOD
// TEX3DLOD: CUDA API:
// TEX3DLOD-NEXT:   tex3DLod<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/,
// TEX3DLOD-NEXT:              f3 /*float*/, f4 /*float*/);
// TEX3DLOD-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// TEX3DLOD-NEXT:   sycl::ext::oneapi::experimental::sample_mipmap<T>(t, sycl::float3(f1, f2, f3), f4);

/// Time Function

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=clock | FileCheck %s -check-prefix=CLOCK
// CLOCK: CUDA API:
// CLOCK-NEXT:   clock();
// CLOCK-NEXT: Is migrated to:
// CLOCK-NEXT:   /*
// CLOCK-NEXT:   DPCT1008:0: clock function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
// CLOCK-NEXT:   */
// CLOCK-NEXT:   clock();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=clock64 | FileCheck %s -check-prefix=CLOCK64
// CLOCK64: CUDA API:
// CLOCK64-NEXT:   clock64();
// CLOCK64-NEXT: Is migrated to:
// CLOCK64-NEXT:   /*
// CLOCK64-NEXT:   DPCT1008:0: clock64 function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
// CLOCK64-NEXT:   */
// CLOCK64-NEXT:   clock64();

/// Atomic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicAdd_system | FileCheck %s -check-prefix=ATOMICADD_SYSTEM
// ATOMICADD_SYSTEM: CUDA API:
// ATOMICADD_SYSTEM-NEXT:   atomicAdd_system(pi /*int **/, i /*int*/);
// ATOMICADD_SYSTEM-NEXT:   atomicAdd_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICADD_SYSTEM-NEXT:   atomicAdd_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICADD_SYSTEM-NEXT:   atomicAdd_system(pf /*float **/, f /*float*/);
// ATOMICADD_SYSTEM-NEXT:   atomicAdd_system(pd /*double **/, d /*double*/);
// ATOMICADD_SYSTEM-NEXT: Is migrated to:
// ATOMICADD_SYSTEM-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICADD_SYSTEM-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICADD_SYSTEM-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);
// ATOMICADD_SYSTEM-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pf, f);
// ATOMICADD_SYSTEM-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pd, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicSub | FileCheck %s -check-prefix=ATOMICSUB
// ATOMICSUB: CUDA API:
// ATOMICSUB-NEXT:   atomicSub(pi /*int **/, i /*int*/);
// ATOMICSUB-NEXT:   atomicSub(pu /*unsigned **/, u /*unsigned*/);
// ATOMICSUB-NEXT: Is migrated to:
// ATOMICSUB-NEXT:   dpct::atomic_fetch_sub<sycl::access::address_space::generic_space>(pi, i);
// ATOMICSUB-NEXT:   dpct::atomic_fetch_sub<sycl::access::address_space::generic_space>(pu, u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicExch_system | FileCheck %s -check-prefix=ATOMICEXCH_SYSTEM
// ATOMICEXCH_SYSTEM: CUDA API:
// ATOMICEXCH_SYSTEM-NEXT:   atomicExch_system(pi /*int **/, i /*int*/);
// ATOMICEXCH_SYSTEM-NEXT:   atomicExch_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICEXCH_SYSTEM-NEXT:   atomicExch_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICEXCH_SYSTEM-NEXT:   atomicExch_system(pf /*float **/, f /*float*/);
// ATOMICEXCH_SYSTEM-NEXT: Is migrated to:
// ATOMICEXCH_SYSTEM-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICEXCH_SYSTEM-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICEXCH_SYSTEM-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);
// ATOMICEXCH_SYSTEM-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pf, f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicExch | FileCheck %s -check-prefix=ATOMICEXCH
// ATOMICEXCH: CUDA API:
// ATOMICEXCH-NEXT:   atomicExch(pi /*int **/, i /*int*/);
// ATOMICEXCH-NEXT:   atomicExch(pu /*unsigned **/, u /*unsigned*/);
// ATOMICEXCH-NEXT:   atomicExch(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICEXCH-NEXT:   atomicExch(pf /*float **/, f /*float*/);
// ATOMICEXCH-NEXT: Is migrated to:
// ATOMICEXCH-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space>(pi, i);
// ATOMICEXCH-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space>(pu, u);
// ATOMICEXCH-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space>(pull, ull);
// ATOMICEXCH-NEXT:   dpct::atomic_exchange<sycl::access::address_space::generic_space>(pf, f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicMin_system | FileCheck %s -check-prefix=ATOMICMIN_SYSTEM
// ATOMICMIN_SYSTEM: CUDA API:
// ATOMICMIN_SYSTEM-NEXT:   atomicMin_system(pi /*int **/, i /*int*/);
// ATOMICMIN_SYSTEM-NEXT:   atomicMin_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICMIN_SYSTEM-NEXT:   atomicMin_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICMIN_SYSTEM-NEXT:   atomicMin_system(pll /*long long **/, ll /*long long*/);
// ATOMICMIN_SYSTEM-NEXT: Is migrated to:
// ATOMICMIN_SYSTEM-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICMIN_SYSTEM-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICMIN_SYSTEM-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);
// ATOMICMIN_SYSTEM-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pll, ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicMin | FileCheck %s -check-prefix=ATOMICMIN
// ATOMICMIN: CUDA API:
// ATOMICMIN-NEXT:   atomicMin(pi /*int **/, i /*int*/);
// ATOMICMIN-NEXT:   atomicMin(pu /*unsigned **/, u /*unsigned*/);
// ATOMICMIN-NEXT:   atomicMin(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICMIN-NEXT:   atomicMin(pll /*long long **/, ll /*long long*/);
// ATOMICMIN-NEXT: Is migrated to:
// ATOMICMIN-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(pi, i);
// ATOMICMIN-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(pu, u);
// ATOMICMIN-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(pull, ull);
// ATOMICMIN-NEXT:   dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(pll, ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicMax_system | FileCheck %s -check-prefix=ATOMICMAX_SYSTEM
// ATOMICMAX_SYSTEM: CUDA API:
// ATOMICMAX_SYSTEM-NEXT:   atomicMax_system(pi /*int **/, i /*int*/);
// ATOMICMAX_SYSTEM-NEXT:   atomicMax_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICMAX_SYSTEM-NEXT:   atomicMax_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICMAX_SYSTEM-NEXT:   atomicMax_system(pll /*long long **/, ll /*long long*/);
// ATOMICMAX_SYSTEM-NEXT: Is migrated to:
// ATOMICMAX_SYSTEM-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICMAX_SYSTEM-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICMAX_SYSTEM-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);
// ATOMICMAX_SYSTEM-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pll, ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicMax | FileCheck %s -check-prefix=ATOMICMAX
// ATOMICMAX: CUDA API:
// ATOMICMAX-NEXT:   atomicMax(pi /*int **/, i /*int*/);
// ATOMICMAX-NEXT:   atomicMax(pu /*unsigned **/, u /*unsigned*/);
// ATOMICMAX-NEXT:   atomicMax(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICMAX-NEXT:   atomicMax(pll /*long long **/, ll /*long long*/);
// ATOMICMAX-NEXT: Is migrated to:
// ATOMICMAX-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(pi, i);
// ATOMICMAX-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(pu, u);
// ATOMICMAX-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(pull, ull);
// ATOMICMAX-NEXT:   dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(pll, ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicInc_system | FileCheck %s -check-prefix=ATOMICINC_SYSTEM
// ATOMICINC_SYSTEM: CUDA API:
// ATOMICINC_SYSTEM-NEXT:   atomicInc_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICINC_SYSTEM-NEXT: Is migrated to:
// ATOMICINC_SYSTEM-NEXT:   dpct::atomic_fetch_compare_inc<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicInc | FileCheck %s -check-prefix=ATOMICINC
// ATOMICINC: CUDA API:
// ATOMICINC-NEXT:   atomicInc(pu /*unsigned **/, u /*unsigned*/);
// ATOMICINC-NEXT: Is migrated to:
// ATOMICINC-NEXT:   dpct::atomic_fetch_compare_inc<sycl::access::address_space::generic_space>(pu, u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicDec_system | FileCheck %s -check-prefix=ATOMICDEC_SYSTEM
// ATOMICDEC_SYSTEM: CUDA API:
// ATOMICDEC_SYSTEM-NEXT:   atomicDec_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICDEC_SYSTEM-NEXT: Is migrated to:
// ATOMICDEC_SYSTEM-NEXT:   dpct::atomic_fetch_compare_dec<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicDec | FileCheck %s -check-prefix=ATOMICDEC
// ATOMICDEC: CUDA API:
// ATOMICDEC-NEXT:   atomicDec(pu /*unsigned **/, u /*unsigned*/);
// ATOMICDEC-NEXT: Is migrated to:
// ATOMICDEC-NEXT:   dpct::atomic_fetch_compare_dec<sycl::access::address_space::generic_space>(pu, u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicCAS_system | FileCheck %s -check-prefix=ATOMICCAS_SYSTEM
// ATOMICCAS_SYSTEM: CUDA API:
// ATOMICCAS_SYSTEM-NEXT:   atomicCAS_system(pi /*int **/, i1 /*int*/, i2 /*int*/);
// ATOMICCAS_SYSTEM-NEXT:   atomicCAS_system(pu /*unsigned **/, u1 /*unsigned*/, u2 /*unsigned*/);
// ATOMICCAS_SYSTEM-NEXT:   atomicCAS_system(pull /*unsigned long long **/, ull1 /*unsigned long long*/,
// ATOMICCAS_SYSTEM-NEXT:             ull2 /*unsigned long long*/);
// ATOMICCAS_SYSTEM-NEXT: Is migrated to:
// ATOMICCAS_SYSTEM-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i1, i2);
// ATOMICCAS_SYSTEM-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u1, u2);
// ATOMICCAS_SYSTEM-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicCAS | FileCheck %s -check-prefix=ATOMICCAS
// ATOMICCAS: CUDA API:
// ATOMICCAS-NEXT:   atomicCAS(pi /*int **/, i1 /*int*/, i2 /*int*/);
// ATOMICCAS-NEXT:   atomicCAS(pu /*unsigned **/, u1 /*unsigned*/, u2 /*unsigned*/);
// ATOMICCAS-NEXT:   atomicCAS(pull /*unsigned long long **/, ull1 /*unsigned long long*/,
// ATOMICCAS-NEXT:             ull2 /*unsigned long long*/);
// ATOMICCAS-NEXT: Is migrated to:
// ATOMICCAS-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(pi, i1, i2);
// ATOMICCAS-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(pu, u1, u2);
// ATOMICCAS-NEXT:   dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(pull, ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicAnd_system | FileCheck %s -check-prefix=ATOMICAND_SYSTEM
// ATOMICAND_SYSTEM: CUDA API:
// ATOMICAND_SYSTEM-NEXT:   atomicAnd_system(pi /*int **/, i /*int*/);
// ATOMICAND_SYSTEM-NEXT:   atomicAnd_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICAND_SYSTEM-NEXT:   atomicAnd_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICAND_SYSTEM-NEXT: Is migrated to:
// ATOMICAND_SYSTEM-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICAND_SYSTEM-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICAND_SYSTEM-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicAnd | FileCheck %s -check-prefix=ATOMICAND
// ATOMICAND: CUDA API:
// ATOMICAND-NEXT:   atomicAnd(pi /*int **/, i /*int*/);
// ATOMICAND-NEXT:   atomicAnd(pu /*unsigned **/, u /*unsigned*/);
// ATOMICAND-NEXT:   atomicAnd(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICAND-NEXT: Is migrated to:
// ATOMICAND-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space>(pi, i);
// ATOMICAND-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space>(pu, u);
// ATOMICAND-NEXT:   dpct::atomic_fetch_and<sycl::access::address_space::generic_space>(pull, ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicOr_system | FileCheck %s -check-prefix=ATOMICOR_SYSTEM
// ATOMICOR_SYSTEM: CUDA API:
// ATOMICOR_SYSTEM-NEXT:   atomicOr_system(pi /*int **/, i /*int*/);
// ATOMICOR_SYSTEM-NEXT:   atomicOr_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICOR_SYSTEM-NEXT:   atomicOr_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICOR_SYSTEM-NEXT: Is migrated to:
// ATOMICOR_SYSTEM-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICOR_SYSTEM-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICOR_SYSTEM-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicOr | FileCheck %s -check-prefix=ATOMICOR
// ATOMICOR: CUDA API:
// ATOMICOR-NEXT:   atomicOr(pi /*int **/, i /*int*/);
// ATOMICOR-NEXT:   atomicOr(pu /*unsigned **/, u /*unsigned*/);
// ATOMICOR-NEXT:   atomicOr(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICOR-NEXT: Is migrated to:
// ATOMICOR-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(pi, i);
// ATOMICOR-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(pu, u);
// ATOMICOR-NEXT:   dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(pull, ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicXor_system | FileCheck %s -check-prefix=ATOMICXOR_SYSTEM
// ATOMICXOR_SYSTEM: CUDA API:
// ATOMICXOR_SYSTEM-NEXT:   atomicXor_system(pi /*int **/, i /*int*/);
// ATOMICXOR_SYSTEM-NEXT:   atomicXor_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICXOR_SYSTEM-NEXT:   atomicXor_system(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICXOR_SYSTEM-NEXT: Is migrated to:
// ATOMICXOR_SYSTEM-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICXOR_SYSTEM-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
// ATOMICXOR_SYSTEM-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pull, ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicXor | FileCheck %s -check-prefix=ATOMICXOR
// ATOMICXOR: CUDA API:
// ATOMICXOR-NEXT:   atomicXor(pi /*int **/, i /*int*/);
// ATOMICXOR-NEXT:   atomicXor(pu /*unsigned **/, u /*unsigned*/);
// ATOMICXOR-NEXT:   atomicXor(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICXOR-NEXT: Is migrated to:
// ATOMICXOR-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space>(pi, i);
// ATOMICXOR-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space>(pu, u);
// ATOMICXOR-NEXT:   dpct::atomic_fetch_xor<sycl::access::address_space::generic_space>(pull, ull);
