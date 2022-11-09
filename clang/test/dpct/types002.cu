// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/types002 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck %s --match-full-lines --input-file %T/types002/types002.dp.cpp

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char **argv) {
  //CHECK:dpct::device_vector<int> device_vec;
  //CHECK-NEXT:int a = sizeof(dpct::device_vector<int>);
  //CHECK-NEXT:a = sizeof(device_vec);
  //CHECK-NEXT:a = sizeof device_vec;
  thrust::device_vector<int> device_vec;
  int a = sizeof(thrust::device_vector<int>);
  a = sizeof(device_vec);
  a = sizeof device_vec;

  //CHECK:dpct::device_pointer<int> device_p;
  //CHECK-NEXT:a = sizeof(dpct::device_pointer<int>);
  //CHECK-NEXT:a = sizeof(device_p);
  //CHECK-NEXT:a = sizeof device_p;
  thrust::device_ptr<int> device_p;
  a = sizeof(thrust::device_ptr<int>);
  a = sizeof(device_p);
  a = sizeof device_p;

  //CHECK:dpct::device_reference<int> device_ref = device_vec[0];
  //CHECK-NEXT:a = sizeof(dpct::device_reference<int>);
  //CHECK-NEXT:a = sizeof(device_ref);
  //CHECK-NEXT:a = sizeof device_ref;
  thrust::device_reference<int> device_ref = device_vec[0];
  a = sizeof(thrust::device_reference<int>);
  a = sizeof(device_ref);
  a = sizeof device_ref;

  //CHECK:std::vector<int> host_vec;
  //CHECK-NEXT:a = sizeof(std::vector<int>);
  //CHECK-NEXT:a = sizeof(host_vec);
  //CHECK-NEXT:a = sizeof host_vec;
  std::vector<int> host_vec;
  a = sizeof(std::vector<int>);
  a = sizeof(host_vec);
  a = sizeof host_vec;
}

template <typename T>
struct alloc {
  // CHECK:      typedef dpct::device_pointer<T> pointer;
  // CHECK-NEXT: typedef dpct::device_pointer<const T> const_pointer;
  // CHECK-NEXT: typedef dpct::device_reference<T> reference;
  // CHECK-NEXT: typedef dpct::device_reference<const T> const_reference;
  typedef thrust::device_ptr<T> pointer;
  typedef thrust::device_ptr<const T> const_pointer;
  typedef thrust::device_reference<T> reference;
  typedef thrust::device_reference<const T> const_reference;
};

template <typename type>
struct bar
{
    enum {
        temp         = 0
    };
};

template <bool IF, typename t1, typename t2>
struct If
{
    typedef t1 t3;
};

// CHECK: template <typename T>
// CHECK-NEXT: struct foo_1
// CHECK-NEXT: {
// CHECK-NEXT:     typedef typename If<bar<sycl::longlong2>::temp, int,int>::t3 Word;
// CHECK-NEXT: };
template <typename T>
struct foo_1
{
    typedef typename If<bar<longlong2>::temp, int,int>::t3 Word;
};

// CHECK: template <typename T>
// CHECK-NEXT: struct foo_2
// CHECK-NEXT: {
// CHECK-NEXT:     typedef typename If<bar<
// CHECK-NEXT:     sycl::longlong2>::temp, int,int>::t3 Word;
// CHECK-NEXT: };
template <typename T>
struct foo_2
{
    typedef typename If<bar<
    longlong2>::temp, int,int>::t3 Word;
};

template<typename foo>
struct type2;

// CHECK: template<>
// CHECK-NEXT: struct type2<float> { typedef sycl::float2  type; };
// CHECK-NEXT: typedef typename type2<float>::type  foo_type;
template<>
struct type2<float> { typedef float2  type; };
typedef typename type2<float>::type  foo_type;

