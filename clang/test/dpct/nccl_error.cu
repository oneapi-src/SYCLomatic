// RUN: dpct --format-range=none -out-root %T/nccl_error %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nccl_error/nccl_error.dp.cpp
// CHECK: #include <dpct/ccl_utils.hpp>
#include "nccl.h"

int main(){
  int version;
  // CHECK: int res;
  ncclResult_t res;

  ncclComm_t comm;
  // CHECK: res = DPCT_CHECK_ERROR(version = dpct::ccl::get_version());
  res = ncclGetVersion(&version);
  // CHECK: int atype = int(oneapi::ccl::datatype::int32);
  int atype = ncclInt32;
  // CHECK: uint16_t uatype = uint16_t(oneapi::ccl::datatype::int32);
  uint16_t uatype = ncclInt32;
  // CHECK: long latype = long(oneapi::ccl::datatype::int32);
  long latype = ncclInt32;
  // CHECK: oneapi::ccl::datatype catype = oneapi::ccl::datatype::int32;
  ncclDataType_t catype = ncclInt32;

  // CHECK: catype = oneapi::ccl::datatype::uint8;
  catype = ncclUint8;
  // CHECK: catype = {oneapi::ccl::datatype::uint8};
  catype = {ncclUint8};
  // CHECK: catype = (oneapi::ccl::datatype::uint8);
  catype = (ncclUint8);
  // CHECK: catype = std::forward<oneapi::ccl::datatype&&>(oneapi::ccl::datatype::uint8);
  catype = std::forward<ncclDataType_t&&>(ncclUint8);
  // CHECK: catype = std::move(oneapi::ccl::datatype::uint8);
  catype = std::move(ncclUint8);

  switch(atype) {        
       // CHECK: case int(oneapi::ccl::datatype::int32): std::cout << "Int32" << std::endl; break;
      case ncclInt32: std::cout << "Int32" << std::endl; break;
  }
  // CHECK:     /*
  // CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: "ncclGetErrorString is not supported"/*ncclGetErrorString(res)*/;
  ncclGetErrorString(res);
  // CHECK:     /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to ncclGetLastError was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  ncclGetLastError(NULL);
  // CHECK:     /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to ncclCommGetAsyncError was removed because this call is redundant in SYCL.
  // CHECK-NEXT: */
  ncclCommGetAsyncError(comm,&res);
  // CHECK: if (res == 0) {
  if (res == ncclSuccess) {
    return 0;
  }
}
