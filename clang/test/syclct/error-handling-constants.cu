// RUN: syclct -out-root %T %s -passes "ErrorConstantsRule" -- -x cuda --cuda-host-only
// RUN: sed -e 's,//.*$,,' %T/error-handling-constants.sycl.cpp | FileCheck --match-full-lines %s

// CHECK:const char *switch_test(cudaError_t error)
// CHECK-NEXT:{
// CHECK-NEXT:    switch (error)
// CHECK-NEXT:    {
// CHECK-NEXT:        case 0:
// CHECK-NEXT:            return "cudaSuccess";
// CHECK-NEXT:        case 1:
// CHECK-NEXT:            return "cudaErrorMissingConfiguration";
// CHECK-NEXT:        case 2:
// CHECK-NEXT:            return "cudaErrorMemoryAllocation";
// CHECK-NEXT:        case 3:
// CHECK-NEXT:            return "cudaErrorInitializationError";
// CHECK-NEXT:        case 4:
// CHECK-NEXT:            return "cudaErrorLaunchFailure";
// CHECK-NEXT:        case 5:
// CHECK-NEXT:            return "cudaErrorPriorLaunchFailure";
// CHECK-NEXT:        case 6:
// CHECK-NEXT:            return "cudaErrorLaunchTimeout";
// CHECK-NEXT:        case 7:
// CHECK-NEXT:            return "cudaErrorLaunchOutOfResources";
// CHECK-NEXT:        case 8:
// CHECK-NEXT:            return "cudaErrorInvalidDeviceFunction";
// CHECK-NEXT:        case 9:
// CHECK-NEXT:            return "cudaErrorInvalidConfiguration";
// CHECK-NEXT:        case 10:
// CHECK-NEXT:            return "cudaErrorInvalidDevice";
// CHECK-NEXT:        case 11:
// CHECK-NEXT:            return "cudaErrorInvalidValue";
// CHECK-NEXT:        case 12:
// CHECK-NEXT:            return "cudaErrorInvalidPitchValue";
// CHECK-NEXT:        case 13:
// CHECK-NEXT:            return "cudaErrorInvalidSymbol";
// CHECK-NEXT:    }
// CHECK-NEXT:    return 0;
// CHECK-NEXT:}
const char *switch_test(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";
        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";
    }
    return 0;
}

// CHECK:int test_simple_ifs() {
// CHECK-NEXT:  cudaError_t err = 13;
// CHECK-NEXT:  if (err != 0) {
// CHECK-NEXT:  }
// CHECK-NEXT:  if (switch_test(1)) {
// CHECK-NEXT:  }
// CHECK-NEXT:  if (0 != err) {
// CHECK-NEXT:  }
// CHECK-NEXT:  if (err == 0) {
// CHECK-NEXT:    return (int)0;
// CHECK-NEXT:  }
// CHECK-NEXT:}
int test_simple_ifs() {
  cudaError_t err = cudaErrorInvalidSymbol;
  if (err != cudaSuccess) {
  }
  if (switch_test(cudaErrorMissingConfiguration)) {
  }
  if (cudaSuccess != err) {
  }
  if (err == cudaSuccess) {
    return (int)cudaSuccess;
  }
}
