// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamBeginCapture | FileCheck %s -check-prefix=CUDASTREAMBEGINCAPTURE
// CUDASTREAMBEGINCAPTURE: CUDA API:
// CUDASTREAMBEGINCAPTURE-NEXT:   cudaStreamBeginCapture(s /*cudaStream_t*/, sc /*cudaStreamCaptureMode*/);
// CUDASTREAMBEGINCAPTURE-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMBEGINCAPTURE-NEXT: dpct::experimental::begin_recording(s);
