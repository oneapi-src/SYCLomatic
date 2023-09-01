// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreate | FileCheck %s -check-prefix=cudnnCreate
// cudnnCreate: CUDA API:
// cudnnCreate-NEXT: cudnnHandle_t h;
// cudnnCreate-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnCreate-NEXT: Is migrated to:
// cudnnCreate-NEXT:   dpct::dnnl::engine_ext h;
// cudnnCreate-NEXT:   h.create_engine();
// cudnnCreate-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnActivationBackward | FileCheck %s -check-prefix=cudnnActivationBackward
// cudnnActivationBackward: CUDA API:
// cudnnActivationBackward-NEXT: cudnnHandle_t h;
// cudnnActivationBackward-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnActivationBackward-NEXT: cudnnActivationBackward(
// cudnnActivationBackward-NEXT:     h /*cudnnHandle_t*/, desc /*cudnnActivationDescriptor_t*/,
// cudnnActivationBackward-NEXT:     alpha /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnActivationBackward-NEXT:     diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnActivationBackward-NEXT:     src_d /*cudnnTensorDescriptor_t*/, src /*void **/, beta /*void **/,
// cudnnActivationBackward-NEXT:     diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
// cudnnActivationBackward-NEXT: Is migrated to:
// cudnnActivationBackward-NEXT: dpct::dnnl::engine_ext h;
// cudnnActivationBackward-NEXT: h.create_engine();
// cudnnActivationBackward-NEXT: h.async_activation_backward(desc, *alpha, dst_d, dst, diff_dst_d, diff_dst, src_d, src, *beta, diff_src_d, diff_src);
// cudnnActivationBackward-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnActivationForward | FileCheck %s -check-prefix=cudnnActivationForward
// cudnnActivationForward: CUDA API:
// cudnnActivationForward-NEXT: cudnnHandle_t h;
// cudnnActivationForward-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnActivationForward-NEXT: cudnnActivationForward(
// cudnnActivationForward-NEXT:     h /*cudnnHandle_t*/, desc /*cudnnActivationDescriptor_t*/,
// cudnnActivationForward-NEXT:     alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnActivationForward-NEXT:     beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnActivationForward-NEXT: Is migrated to:
// cudnnActivationForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnActivationForward-NEXT:   h.create_engine();
// cudnnActivationForward-NEXT:   h.async_activation_forward(desc, *alpha, src_d, src, *beta, dst_d, dst);
// cudnnActivationForward-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateActivationDescriptor | FileCheck %s -check-prefix=cudnnCreateActivationDescriptor
// cudnnCreateActivationDescriptor: CUDA API:
// cudnnCreateActivationDescriptor-NEXT:   cudnnCreateActivationDescriptor(d /*cudnnActivationDescriptor_t **/);
// cudnnCreateActivationDescriptor-NEXT: The API is Removed.
// cudnnCreateActivationDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateTensorDescriptor | FileCheck %s -check-prefix=cudnnCreateTensorDescriptor
// cudnnCreateTensorDescriptor: CUDA API:
// cudnnCreateTensorDescriptor-NEXT:   cudnnCreateTensorDescriptor(d /*cudnnTensorDescriptor_t **/);
// cudnnCreateTensorDescriptor-NEXT: The API is Removed.
// cudnnCreateTensorDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroy | FileCheck %s -check-prefix=cudnnDestroy
// cudnnDestroy: CUDA API:
// cudnnDestroy-NEXT: cudnnDestroy(h /*cudnnHandle_t*/);
// cudnnDestroy-NEXT: The API is Removed.
// cudnnDestroy-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetActivationDescriptor | FileCheck %s -check-prefix=cudnnSetActivationDescriptor
// cudnnSetActivationDescriptor: CUDA API:
// cudnnSetActivationDescriptor-NEXT: cudnnActivationDescriptor_t d;
// cudnnSetActivationDescriptor-NEXT: cudnnSetActivationDescriptor(d /*cudnnActivationDescriptor_t*/,
// cudnnSetActivationDescriptor-NEXT:                              m /*cudnnActivationMode_t*/,
// cudnnSetActivationDescriptor-NEXT:                              p /*cudnnNanPropagation_t*/, c /*double*/);
// cudnnSetActivationDescriptor-NEXT: Is migrated to:
// cudnnSetActivationDescriptor-NEXT:   dpct::dnnl::activation_desc d;
// cudnnSetActivationDescriptor-NEXT:   d.set(m, c);
// cudnnSetActivationDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetStream | FileCheck %s -check-prefix=cudnnSetStream
// cudnnSetStream: CUDA API:
// cudnnSetStream-NEXT: cudnnHandle_t h;
// cudnnSetStream-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnSetStream-NEXT: cudnnSetStream(h /*cudnnHandle_t*/, s /*cudaStream_t*/);
// cudnnSetStream-NEXT: Is migrated to:
// cudnnSetStream-NEXT:   dpct::dnnl::engine_ext h;
// cudnnSetStream-NEXT:   h.create_engine();
// cudnnSetStream-NEXT:   h.set_queue(s);
// cudnnSetStream-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensor4dDescriptor | FileCheck %s -check-prefix=cudnnSetTensor4dDescriptor
// cudnnSetTensor4dDescriptor: CUDA API:
// cudnnSetTensor4dDescriptor-NEXT: cudnnTensorDescriptor_t d;
// cudnnSetTensor4dDescriptor-NEXT: cudnnSetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensor4dDescriptor-NEXT:                            f /*cudnnTensorFormat_t*/, t /*cudnnDataType_t*/,
// cudnnSetTensor4dDescriptor-NEXT:                            n /*int*/, c /*int*/, h /*int*/, w /*int*/);
// cudnnSetTensor4dDescriptor-NEXT: Is migrated to:
// cudnnSetTensor4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetTensor4dDescriptor-NEXT:   d.set(f, t, n, c, h, w);
// cudnnSetTensor4dDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetTensor4dDescriptor | FileCheck %s -check-prefix=cudnnGetTensor4dDescriptor
// cudnnGetTensor4dDescriptor: CUDA API:
// cudnnGetTensor4dDescriptor-NEXT: cudnnTensorDescriptor_t d;
// cudnnGetTensor4dDescriptor-NEXT: cudnnGetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnGetTensor4dDescriptor-NEXT:                            t /*cudnnDataType_t **/, n /*int **/, c /*int **/,
// cudnnGetTensor4dDescriptor-NEXT:                            h /*int **/, w /*int **/, ns /*int **/,
// cudnnGetTensor4dDescriptor-NEXT:                            cs /*int **/, hs /*int **/, ws /*int **/);
// cudnnGetTensor4dDescriptor-NEXT: Is migrated to:
// cudnnGetTensor4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetTensor4dDescriptor-NEXT:   d.get(t, n, c, h, w, ns, cs, hs, ws);
// cudnnGetTensor4dDescriptor-EMPTY:
