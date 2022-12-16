// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T/template_instantiation_argment %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/template_instantiation_argment/template_instantiation_argment.dp.cpp

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

template <typename BinaryFunction, typename T>
void BroadcastMatrixVector(
    BinaryFunction binary_operation,
    const T alpha) {
  ;
}

// CHECK: template void BroadcastMatrixVector<std::divides<float>, float>(
// CHECK-NEXT: std::divides<float> binary_operation,
// CHECK-NEXT: const float alpha);
template void BroadcastMatrixVector<thrust::divides<float>, float>(
    thrust::divides<float> binary_operation,
    const float alpha);

// CHECK: template void BroadcastMatrixVector<std::minus<float>, float>(
// CHECK-NEXT: std::minus<float> binary_operation,
// CHECK-NEXT: const float alpha);
template void BroadcastMatrixVector<thrust::minus<float>, float>(
    thrust::minus<float> binary_operation,
    const float alpha);

// CHECK: template void BroadcastMatrixVector<std::multiplies<std::complex<float>>, std::complex<float>>(
// CHECK-NEXT: std::multiplies<std::complex<float>> binary_operation,
// CHECK-NEXT: const std::complex<float> alpha);
template void BroadcastMatrixVector<thrust::multiplies<thrust::complex<float>>, thrust::complex<float>>(
    thrust::multiplies<thrust::complex<float>> binary_operation,
    const thrust::complex<float> alpha);
