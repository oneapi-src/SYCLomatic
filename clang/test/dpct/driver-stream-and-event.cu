// RUN: dpct -out-root %T/driver-stream-and-event %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-stream-and-event/driver-stream-and-event.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/driver-stream-and-event/driver-stream-and-event.dp.cpp -o %T/driver-stream-and-event/driver-stream-and-event.dp.o %}
#include "cuda.h"
#include <vector>
// CHECK: #include <future>
template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

void foo(){
  CUfunction f;
  CUstream s;
  CUevent e;

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuFuncSetCacheConfig was removed because SYCL currently does not support configuring shared memory on devices.
  //CHECK-NEXT: */
  cuFuncSetCacheConfig(f, CU_FUNC_CACHE_PREFER_NONE);

  //CHECK: s = dpct::get_current_device().create_queue();
  cuStreamCreate(&s, CU_STREAM_DEFAULT);

  // CHECK: int streamStatus = DPCT_CHECK_ERROR((s->ext_oneapi_empty())); 
  // CHECK-NEXT: if (streamStatus == 0);
  CUresult streamStatus = cuStreamQuery(s);
  if (streamStatus == CUDA_SUCCESS);

  //CHECK: s->wait();
  cuStreamSynchronize(s);

  // CHECK: if (DPCT_CHECK_ERROR((s->ext_oneapi_empty())) == 0);
  if (cuStreamQuery(s) == CUDA_SUCCESS);

  //CHECK: s->ext_oneapi_submit_barrier({*e});
  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

  //CHECK: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: e_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: *e = s->ext_oneapi_submit_barrier();
  //CHECK-NEXT: e->wait_and_throw();
  cuEventRecord(e, s);
  cuEventSynchronize(e);

  //CHECK: sycl::info::event_command_status r;
  //CHECK-NEXT: r = e->get_info<sycl::info::event::command_execution_status>();
  CUresult r;
  r = cuEventQuery(e);

  //CHECK: dpct::event_ptr start, end;
  //CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  //CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> end_ct1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: start_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: end_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: float result_time;
  //CHECK-NEXT: result_time = std::chrono::duration<float, std::milli>(end_ct1 - start_ct1).count();
  CUevent start, end;
  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);

  int rr;
  //CHECK: rr = dpct::get_kernel_function_info(f).max_work_group_size;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);

  //CHECK: /*
  //CHECK-NEXT: DPCT1132:{{[0-9]+}}: SYCL 2020 does not support accessing the statically allocated shared memory for the kernel. The API is replaced with member variable "shared_size_bytes". Please set the appropriate value for "shared_size_bytes".
  //CHECK-NEXT: */
  //CHECK: rr = dpct::get_kernel_function_info(f).shared_size_bytes /* statically allocated shared memory per work-group in bytes */;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);

  //CHECK: /*
  //CHECK-NEXT: DPCT1132:{{[0-9]+}}: SYCL 2020 does not support accessing the local memory for the kernel. The API is replaced with member variable "local_size_bytes". Please set the appropriate value for "local_size_bytes".
  //CHECK-NEXT: */
  //CHECK: rr = dpct::get_kernel_function_info(f).local_size_bytes /* local memory per work-item in bytes */;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);

  //CHECK: /*
  //CHECK-NEXT: DPCT1132:{{[0-9]+}}: SYCL 2020 does not support accessing the memory size of user-defined constants for the kernel. The API is replaced with member variable "const_size_bytes". Please set the appropriate value for "const_size_bytes".
  //CHECK-NEXT: */
  //CHECK: rr = dpct::get_kernel_function_info(f).const_size_bytes /* user-defined constant kernel memory in bytes */;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, f);

  //CHECK: /*
  //CHECK-NEXT: DPCT1132:{{[0-9]+}}: SYCL 2020 does not support accessing the required number of registers for the kernel. The API is replaced with member variable "num_regs". Please set the appropriate value for "num_regs".
  //CHECK-NEXT: */
  //CHECK: rr = dpct::get_kernel_function_info(f).num_regs /* number of registers for each thread */;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_NUM_REGS, f);

  cuEventDestroy(start);

  cuEventDestroy(end);
}


// CHECK: void process(dpct::queue_ptr st, char *data, int status) {}
void process(CUstream st, char *data, CUresult status) {}

template<typename T>
// CHECK: void callback(dpct::queue_ptr hStream, int status, void *userData) {
void callback(CUstream hStream, CUresult status, void* userData) {
  T *data = static_cast<T *>(userData);
  process(hStream, data, status);
}

void test_stream() {
  CUstream hStream;
  void* data;
  unsigned int flag;
  size_t length;
  CUdeviceptr  cuPtr;
  // CHECK: std::async([&]() { hStream->wait(); callback<char>(hStream, 0, data); });
  cuStreamAddCallback(hStream, callback<char>, data, flag);

  // CHECK: int result = DPCT_CHECK_ERROR(std::async([&]() { hStream->wait(); callback<char>(hStream, 0, data); }));
  CUresult result = cuStreamAddCallback(hStream, callback<char>, data, flag);

  // CHECK dpct::queue_callback cbptr = callback<char>;
  CUstreamCallback cbptr = callback<char>;
  // CHECK: std::async([&]() { hStream->wait(); cbptr(hStream, 0, data); });
  cuStreamAddCallback(hStream, cbptr, data, flag);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuStreamAttachMemAsync was removed because SYCL currently does not support associating USM with a specific queue.
  //CHECK-NEXT: */
  cuStreamAttachMemAsync(hStream, cuPtr, length, flag);

  //CHECK: /*
  //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuStreamAttachMemAsync was replaced with 0 because SYCL currently does not support associating USM with a specific queue.
  //CHECK-NEXT: */
  //CHECK-NEXT: result = 0;
  result = cuStreamAttachMemAsync(hStream, cuPtr, length, flag);

  //CHECK: /*
  //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuStreamAttachMemAsync was replaced with 0 because SYCL currently does not support associating USM with a specific queue.
  //CHECK-NEXT: */
  //CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cuStreamAttachMemAsync(hStream, cuPtr, length, flag));

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuStreamAttachMemAsync was removed because SYCL currently does not support associating USM with a specific queue.
  //CHECK-NEXT: */
  cuStreamAttachMemAsync(hStream, cuPtr, std::vector<int>(1,1).front(), flag);

  // CHECK: dpct::get_current_device().destroy_queue(hStream);
  cuStreamDestroy(hStream);
}

void test_cuEventRecord_crash(CUevent hEvent, CUstream hStream)
{
  // CHECK: int result = DPCT_CHECK_ERROR(*(dpct::event_ptr)hEvent = ((dpct::queue_ptr)hStream)->ext_oneapi_submit_barrier());
  CUresult result = cuEventRecord((CUevent)hEvent, (CUstream)hStream);
}

unsigned getEventFlags(bool enabledSyncBlock) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The CU_EVENT_DISABLE_TIMING is not supported for SYCL event. The output parameter(s) is set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: unsigned flags = 0;
  unsigned flags = CU_EVENT_DISABLE_TIMING;

  if (enabledSyncBlock)
    // CHECK: /*
    // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The CU_EVENT_BLOCKING_SYNC is not supported for SYCL event. The output parameter(s) is set to 0.
    // CHECK-NEXT: */
    // CHECK-NEXT: flags |= 0;
    flags |= CU_EVENT_BLOCKING_SYNC;

  return flags;
}
