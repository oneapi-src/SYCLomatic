// RUN: dpct --format-range=none -out-root %T/error-handling-constants %s -passes "ErrorConstantsRule" --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/error-handling-constants/error-handling-constants.dp.cpp --match-full-lines %s


#include <cufft.h>
#include <stdexcept>

// CHECK:const char *switch_test(cudaError_t error)
// CHECK-NEXT:{
// CHECK-NEXT:    switch (error)
// CHECK-NEXT:    {
// CHECK-NEXT:        case 0:
// CHECK-NEXT:            return "cudaSuccess";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorMissingConfiguration";
// CHECK-NEXT:        case 2:
// CHECK-NEXT:            return "cudaErrorMemoryAllocation";
// CHECK-NEXT:        case 3:
// CHECK-NEXT:            return "cudaErrorInitializationError";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchFailure";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorPriorLaunchFailure";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchTimeout";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorLaunchOutOfResources";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorInvalidDeviceFunction";
// CHECK-NEXT:        case 9:
// CHECK-NEXT:            return "cudaErrorInvalidConfiguration";
// CHECK-NEXT:        case {{[0-9]+}}:
// CHECK-NEXT:            return "cudaErrorInvalidDevice";
// CHECK-NEXT:        case {{[0-9]+}}:
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

// CHECK:static void assert_cufft_(cufftResult_t t, const char * msg)
// CHECK-NEXT:{
// CHECK-NEXT:    if (t == 0)
// CHECK-NEXT:        return;
// CHECK-NEXT:    std::string w(msg);
// CHECK-NEXT:    w += ": ";
// CHECK-NEXT:    switch (t)
// CHECK-NEXT:    {
// CHECK-NEXT:        case 1:
// CHECK-NEXT:            w += "CUFFT was passed an invalid plan handle"; break;
// CHECK-NEXT:        case 2:
// CHECK-NEXT:            w += "CUFFT failed to allocate GPU or CPU memory"; break;
// CHECK-NEXT:        case 3:
// CHECK-NEXT:            w += "Unused"; break;
// CHECK-NEXT:        case 4:
// CHECK-NEXT:            w += "User specified an invalid pointer or parameter"; break;
// CHECK-NEXT:        case 5:
// CHECK-NEXT:            w += "Used for all driver and internal CUFFT library errors"; break;
// CHECK-NEXT:        case 6:
// CHECK-NEXT:            w += "CUFFT failed to execute an FFT on the GPU"; break;
// CHECK-NEXT:        case 7:
// CHECK-NEXT:            w += "The CUFFT library failed to initialize"; break;
// CHECK-NEXT:        case 8:
// CHECK-NEXT:            w += "User specified an invalid transform size"; break;
// CHECK-NEXT:        default:
// CHECK-NEXT:            w += "Unknown CUFFT error";
// CHECK-NEXT:    }
// CHECK-NEXT:    throw std::runtime_error(w);
// CHECK-NEXT:}

static void assert_cufft_(cufftResult_t t, const char * msg)
{
    if (t == CUFFT_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    switch (t)
    {
        case CUFFT_INVALID_PLAN:
            w += "CUFFT was passed an invalid plan handle"; break;
        case CUFFT_ALLOC_FAILED:
            w += "CUFFT failed to allocate GPU or CPU memory"; break;
        case CUFFT_INVALID_TYPE:
            w += "Unused"; break;
        case CUFFT_INVALID_VALUE:
            w += "User specified an invalid pointer or parameter"; break;
        case CUFFT_INTERNAL_ERROR:
            w += "Used for all driver and internal CUFFT library errors"; break;
        case CUFFT_EXEC_FAILED:
            w += "CUFFT failed to execute an FFT on the GPU"; break;
        case CUFFT_SETUP_FAILED:
            w += "The CUFFT library failed to initialize"; break;
        case CUFFT_INVALID_SIZE:
            w += "User specified an invalid transform size"; break;
        default:
            w += "Unknown CUFFT error";
    }
    throw std::runtime_error(w);
}

// CHECK:int test_simple_ifs() {
// CHECK-NEXT:  cudaError_t err = 13;
// CHECK-NEXT:  if (err != 0) {
// CHECK-NEXT:  }
// CHECK-NEXT:  if (switch_test({{[0-9]+}})) {
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

