// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0
// RUN: dpct --format-range=none -in-root %S -out-root %T/Libcu %S/libcu_num.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/Libcu/libcu_num.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/Libcu/libcu_num.dp.cpp -o %T/Libcu/libcu_num.dp.o %}

#include <cuda/std/climits>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/tuple>
 
 #include <iostream>
template <typename T>  T init_value() {
    //CHECK: return -std::numeric_limits<T>::infinity();
    return -cuda::std::numeric_limits<T>::infinity();
 }
//CHECK:   template <typename T> constexpr T terminate_value() { return std::numeric_limits<T>::infinity(); }
  template <typename T> constexpr T terminate_value() { return cuda::std::numeric_limits<T>::infinity(); }

  template <class T> bool is_complete(const T &result) { return result != init_value<T>(); }


  int main() {
   std::cout << init_value<int>();  
    return 0;
  }