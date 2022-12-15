// RUN: rm -rf %T/Util/api_test27_out
// RUN: dpct --use-custom-helper=api --out-root=%T/Util/api_test27_out \
// RUN:      --cuda-include-path="%cuda-path/include" \
// RUN:      %s -- -x cuda -ptx
// RUN: grep "int_as_queue_ptr:" %T/Util/api_test27_out/MainSourceFiles.yaml \
// RUN: | python -c "assert len(input().splitlines()) == 1"


// TEST_FEATURE: Util_int_as_queue_ptr

void CopyToHost(void *buf, void *host, int N, int stream) {
  cudaMemcpyAsync(buf, host, N*sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}
