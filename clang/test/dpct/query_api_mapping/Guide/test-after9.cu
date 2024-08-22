// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0

/// Synchronization Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__syncwarp | FileCheck %s -check-prefix=__SYNCWARP
// __SYNCWARP: CUDA API:
// __SYNCWARP-NEXT:   __syncwarp(u /*unsigned*/);
// __SYNCWARP-NEXT: Is migrated to (with the option --use-experimental-features=free-function-queries):
// __SYNCWARP-NEXT:   sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

/// Atomic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicAdd | FileCheck %s -check-prefix=ATOMICADD
// ATOMICADD: CUDA API:
// ATOMICADD-NEXT:   atomicAdd(pi /*int **/, i /*int*/);
// ATOMICADD-NEXT:   atomicAdd(pu /*unsigned **/, u /*unsigned*/);
// ATOMICADD-NEXT:   atomicAdd(pull /*unsigned long long **/, ull /*unsigned long long*/);
// ATOMICADD-NEXT:   atomicAdd(pf /*float **/, f /*float*/);
// ATOMICADD-NEXT:   atomicAdd(pd /*double **/, d /*double*/);
// ATOMICADD-NEXT:   atomicAdd(ph2 /*__half2 **/, h2 /*__half2*/);
// ATOMICADD-NEXT: Is migrated to:
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(pi, i);
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(pu, u);
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(pull, ull);
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(pf, f);
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(pd, d);
// ATOMICADD-NEXT:   dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(ph2, h2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atomicSub_system | FileCheck %s -check-prefix=ATOMICSUB_SYSTEM
// ATOMICSUB_SYSTEM: CUDA API:
// ATOMICSUB_SYSTEM-NEXT:   atomicSub_system(pi /*int **/, i /*int*/);
// ATOMICSUB_SYSTEM-NEXT:   atomicSub_system(pu /*unsigned **/, u /*unsigned*/);
// ATOMICSUB_SYSTEM-NEXT: Is migrated to:
// ATOMICSUB_SYSTEM-NEXT:   dpct::atomic_fetch_sub<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pi, i);
// ATOMICSUB_SYSTEM-NEXT:   dpct::atomic_fetch_sub<sycl::access::address_space::generic_space, sycl::memory_order::relaxed, sycl::memory_scope::system>(pu, u);
