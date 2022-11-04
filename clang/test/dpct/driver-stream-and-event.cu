// RUN: dpct -out-root %T/driver-stream-and-event %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-stream-and-event/driver-stream-and-event.dp.cpp %s

#include<vector>
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
  //CHECK-NEXT: s->wait();
  cuStreamCreate(&s, CU_STREAM_DEFAULT);
  cuStreamSynchronize(s);

  //CHECK: s->ext_oneapi_submit_barrier({*e});
  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

  //CHECK: *e = s->ext_oneapi_submit_barrier();
  //CHECK-NEXT: e->wait_and_throw();
  cuEventRecord(e, s);
  cuEventSynchronize(e);

  //CHECK: sycl::info::event_command_status r;
  //CHECK-NEXT: r = e->get_info<sycl::info::event::command_execution_status>();
  CUresult r;
  r = cuEventQuery(e);

  //CHECK: dpct::event_ptr start, end;
  //CHECK: *start = s->ext_oneapi_submit_barrier();
  //CHECK: *end = s->ext_oneapi_submit_barrier();
  //CHECK: start->wait_and_throw();
  //CHECK: end->wait_and_throw();
  //CHECK: float result_time;
  //CHECK: result_time = (end->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  CUevent start, end;
  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);

  int rr;
  //CHECK: rr = dpct::get_kernel_function_info((const void *)f).max_work_group_size;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);

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
  // CHECK: std::async([&]() {hStream->wait(); callback<char>(hStream, 0, data); });
  cuStreamAddCallback(hStream, callback<char>, data, flag);

  // CHECK: int result = (std::async([&]() {hStream->wait(); callback<char>(hStream, 0, data); }), 0);
  CUresult result = cuStreamAddCallback(hStream, callback<char>, data, flag);

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
