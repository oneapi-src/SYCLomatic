// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamBeginCapture | FileCheck %s -check-prefix=CUDASTREAMBEGINCAPTURE
// CUDASTREAMBEGINCAPTURE: CUDA API:
// CUDASTREAMBEGINCAPTURE-NEXT:   cudaStreamBeginCapture(s /*cudaStream_t*/, sc /*cudaStreamCaptureMode*/);
// CUDASTREAMBEGINCAPTURE-NEXT: The API is Removed.
// CUDASTREAMBEGINCAPTURE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamEndCapture | FileCheck %s -check-prefix=CUDASTREAMENDCAPTURE
// CUDASTREAMENDCAPTURE: CUDA API:
// CUDASTREAMENDCAPTURE-NEXT:   cudaStreamEndCapture(s /*cudaStream_t*/, pg /*cudaGraph_t **/);
// CUDASTREAMENDCAPTURE-NEXT: The API is Removed.
// CUDASTREAMENDCAPTURE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamIsCapturing | FileCheck %s -check-prefix=CUDASTREAMISCAPTURING
// CUDASTREAMISCAPTURING: CUDA API:
// CUDASTREAMISCAPTURING-NEXT:   cudaStreamIsCapturing(s /*cudaStream_t*/,
// CUDASTREAMISCAPTURING-NEXT:                         ps /* enum cudaStreamCaptureStatus **/);
// CUDASTREAMISCAPTURING-NEXT: The API is Removed.
// CUDASTREAMISCAPTURING-EMPTY:
