#include "macro_def.hh"
#include "util_bar.hh"
#include <iostream>

// CHECK:    /*
// CHECK-NEXT:    DPCT1040:{{[0-9]+}}: Use sycl::stream instead of printf if your code is used on the device.
// CHECK-NEXT:    */
#define foo_assert( cond) \
   do \
   { \
      if (!(cond)) \
      { \
        printf("ERROR\n"); \
      } \
   } while(0)

// CHECK:SYCL_EXTERNAL HOST_DEVICE
// CHECK-NEXT:void util_bar(){}
// CHECK-NEXT:HOST_DEVICE_END
HOST_DEVICE
void util_bar(void){}
HOST_DEVICE_END

// CHECK:SYCL_EXTERNAL HOST_DEVICE
// CHECK-NEXT:void FooQueue::
// CHECK-NEXT:push( int neighbor_, int vault_index_ ){
// CHECK-NEXT:    int a = 2;
// CHECK-NEXT:    foo_assert(a > 1);
// CHECK-NEXT:}
HOST_DEVICE
void FooQueue::
push( int neighbor_, int vault_index_ ){
    int a = 2;
    foo_assert(a > 1);
}
HOST_DEVICE_END

// CHECK:SYCL_EXTERNAL HOST_DEVICE
// CHECK-NEXT:void SubFooReaction::fooCollision(){
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1040:{{[0-9]+}}: Use sycl::stream instead of printf if your code is used on the device.
// CHECK-NEXT:    */
// CHECK-NEXT:    printf("hello");
// CHECK-NEXT:}
HOST_DEVICE
void SubFooReaction::fooCollision(){
    printf("hello");
}
HOST_DEVICE_END
