//CHECK: #define THREAD_IDX_X item_ct1.get_local_id(2)
#define THREAD_IDX_X threadIdx.x

#define STRINGIFY_(...) #__VA_ARGS__
#define STRINGIFY(...) STRINGIFY_(__VA_ARGS__)

//CHECK: void foo40_dev_func(const char *file_name, const char *other,
//CHECK-NEXT:                     const sycl::stream &stream_ct1) {
//CHECK-NEXT:   stream_ct1 << "sss\n";
//CHECK-NEXT: }
//CHECK-NEXT: #define FOO40_MACRO                                                            \
//CHECK-NEXT:   foo40_dev_func(__FILE__, STRINGIFY(__OTHER_MACRO__), stream_ct1)
__device__ void foo40_dev_func(const char *file_name, const char *other) {
  printf("sss\n");
}
#define FOO40_MACRO foo40_dev_func(__FILE__, STRINGIFY(__OTHER_MACRO__))
