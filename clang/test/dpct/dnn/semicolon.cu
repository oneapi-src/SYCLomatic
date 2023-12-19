// RUN: dpct %s --out-root %T/semicolon --cuda-include-path="%cuda-path/include" --format-range=none 
// RUN: FileCheck %s --match-full-lines --input-file %T/semicolon/semicolon.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/semicolon/semicolon.dp.cpp -o %T/semicolon/semicolon.dp.o %}
#include <cudnn.h>

int main() {
  // CHECK:      if (true) {
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1026:0: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT:   */
  // CHECK-NEXT: }
  if (true) {
    cudnnDestroy(nullptr);
  }

  // CHECK:      if (true)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   ;
  if (true)
    cudnnDestroy(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateActivationDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateActivationDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyActivationDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyActivationDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateLRNDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateLRNDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyLRNDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyLRNDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreatePoolingDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreatePoolingDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyPoolingDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyPoolingDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateReduceTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateReduceTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyReduceTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyReduceTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateOpTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateOpTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyOpTensorDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyOpTensorDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateFilterDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateFilterDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyFilterDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyFilterDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateConvolutionDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateConvolutionDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyConvolutionDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyConvolutionDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateRNNDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateRNNDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnCreateRNNDataDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnCreateRNNDataDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyRNNDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyRNNDescriptor(nullptr);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyRNNDataDescriptor was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) ;
  if (true) cudnnDestroyRNNDataDescriptor(nullptr);
}
