// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamBeginCapture | FileCheck %s -check-prefix=CUDASTREAMBEGINCAPTURE
// CUDASTREAMBEGINCAPTURE: CUDA API:
// CUDASTREAMBEGINCAPTURE-NEXT:   cudaStreamBeginCapture(s /*cudaStream_t*/, sc /*cudaStreamCaptureMode*/);
// CUDASTREAMBEGINCAPTURE-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMBEGINCAPTURE-NEXT: dpct::experimental::begin_recording(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamEndCapture | FileCheck %s -check-prefix=CUDASTREAMENDCAPTURE
// CUDASTREAMENDCAPTURE: CUDA API:
// CUDASTREAMENDCAPTURE-NEXT:   cudaStreamEndCapture(s /*cudaStream_t*/, pg /*cudaGraph_t **/);
// CUDASTREAMENDCAPTURE-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMENDCAPTURE-NEXT: dpct::experimental::end_recording(s, pg);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamIsCapturing | FileCheck %s -check-prefix=CUDASTREAMISCAPTURING
// CUDASTREAMISCAPTURING: CUDA API:
// CUDASTREAMISCAPTURING-NEXT:   cudaStreamIsCapturing(s /*cudaStream_t*/,
// CUDASTREAMISCAPTURING-NEXT:                         ps /* enum cudaStreamCaptureStatus **/);
// CUDASTREAMISCAPTURING-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// CUDASTREAMISCAPTURING-NEXT: *ps = s->ext_oneapi_get_state();
