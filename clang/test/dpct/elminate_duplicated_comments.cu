// RUN: dpct --format-range=none -out-root %T/elminate_duplicated_comments %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/elminate_duplicated_comments/elminate_duplicated_comments.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/elminate_duplicated_comments/elminate_duplicated_comments.dp.cpp -o %T/elminate_duplicated_comments/elminate_duplicated_comments.dp.o %}

int main() {

  // CHECK: dpct::device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:int hash = deviceProp.get_major_version() ^ deviceProp.get_minor_version();
  int hash = deviceProp.major ^ deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.get_major_version() ^ deviceProp.get_minor_version() ^ deviceProp.get_major_version() ^ deviceProp.get_minor_version();
  hash = deviceProp.major ^ deviceProp.minor ^ deviceProp.major ^ deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.get_major_version() ^
  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:deviceProp.get_minor_version();
  hash = deviceProp.major ^
         deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.get_major_version() ^ deviceProp.get_major_version() ^
  // CHECK:/*
  // CHECK-NEXT:DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:deviceProp.get_minor_version() ^ deviceProp.get_minor_version();
  hash = deviceProp.major ^ deviceProp.major ^
         deviceProp.minor ^ deviceProp.minor;

  return 0;
}

