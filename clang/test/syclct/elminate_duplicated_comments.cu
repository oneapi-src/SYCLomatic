// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/elminate_duplicated_comments.sycl.cpp

int main() {

  // CHECK: syclct::sycl_device_info deviceProp;
  cudaDeviceProp deviceProp;

  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:int hash = deviceProp.major_version() ^ deviceProp.minor_version();
  int hash = deviceProp.major ^ deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.major_version() ^ deviceProp.minor_version() ^ deviceProp.major_version() ^ deviceProp.minor_version();
  hash = deviceProp.major ^ deviceProp.minor ^ deviceProp.major ^ deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.major_version() ^
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:deviceProp.minor_version();
  hash = deviceProp.major ^
         deviceProp.minor;

  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:hash = deviceProp.major_version() ^ deviceProp.major_version() ^
  // CHECK:/*
  // CHECK-NEXT:SYCLCT1005:{{[0-9]+}}: The device version is different. You may want to rewrite this code
  // CHECK-NEXT:*/
  // CHECK-NEXT:deviceProp.minor_version() ^ deviceProp.minor_version();
  hash = deviceProp.major ^ deviceProp.major ^
         deviceProp.minor ^ deviceProp.minor;

  return 0;
}
