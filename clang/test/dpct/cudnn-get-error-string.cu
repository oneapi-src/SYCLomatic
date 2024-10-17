// RUN: dpct --format-range=none -out-root %T/cudnn-get-error-string %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cudnn-get-error-string/cudnn-get-error-string.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cudnn-get-error-string/cudnn-get-error-string.dp.cpp -o %T/cudnn-get-error-string/cudnn-get-error-string.dp.o %}

#include <cudnn.h>

int printf(const char *format, ...);

// CHECK: #define PRINT_ERROR_STR(X) printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR(X) printf("%s\n", cudnnGetErrorString(X))

// CHECK: #define PRINT_ERROR_STR2(X)\
// CHECK-NEXT:  printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR2(X)\
  printf("%s\n", cudnnGetErrorString(X))

// CHECK: #define PRINT_ERROR_STR3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR3(X)\
  printf("%s\
         \n", cudnnGetErrorString(X))


const char *test_function(cudnnStatus_t status) {

//CHECK:  printf("%s\n", dpct::get_error_string_dummy(status));
  printf("%s\n", cudnnGetErrorString(status));

//CHECK:  printf("%s\n", dpct::get_error_string_dummy(0));
  printf("%s\n", cudnnGetErrorString(CUDNN_STATUS_SUCCESS));

  PRINT_ERROR_STR(status);
  PRINT_ERROR_STR2(status);
  PRINT_ERROR_STR3(status);  
}

//CHECK:void foo(dpct::err1 err) {
//CHECK-NEXT:  dpct::get_error_string_dummy(err);
//CHECK-NEXT:  dpct::get_error_string_dummy({{[0-9]+}});
//CHECK-NEXT:}
void foo(cudnnStatus_t err) {
  cudnnGetErrorString(err);
  cudnnGetErrorString(CUDNN_STATUS_NOT_INITIALIZED);
}
