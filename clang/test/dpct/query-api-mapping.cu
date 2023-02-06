// RUN: dpct --query-api-mapping=cudaDeviceGetLimit > output.log
// RUN: FileCheck %s --match-full-lines --input-file output.log -check-prefix=DISABLE

// DISABLE: Mapping for cudaDeviceGetLimit is not available

// CHECK: CUDA API:
// CHECK-NEXT:     cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit)
// CHECK-NEXT: Migrates to SYCL API:
// CHECK-NEXT:     =
// CHECK-NEXT: For example, the CUDA code:
// CHECK-NEXT:     int main() {
// CHECK-NEXT:       size_t *pValue;
// CHECK-NEXT:       cudaLimit limit;
// CHECK-NEXT:       cudaDeviceGetLimit(pValue, limit);
// CHECK-NEXT:     }
// CHECK-NEXT: Is migrated to SYCL code:
// CHECK-NEXT:     #include <sycl/sycl.hpp>
// CHECK-NEXT:     #include <dpct/dpct.hpp>
// CHECK-NEXT:     int main() {
// CHECK-NEXT:       size_t *pValue;
// CHECK-NEXT:       cudaLimit limit;
// CHECK-NEXT:       /*
// CHECK-NEXT:       DPCT1029:0: SYCL currently does not support getting device resource limits.
// CHECK-NEXT:       The output parameter(s) are set to 0.
// CHECK-NEXT:       */
// CHECK-NEXT:       *pValue = 0;
// CHECK-NEXT:     }
