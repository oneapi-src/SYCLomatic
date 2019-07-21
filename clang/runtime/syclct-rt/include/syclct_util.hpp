/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- syclct_util.hpp -------------------------------*- C++ -*-----===//

#ifndef SYCLCT_UTIL_H
#define SYCLCT_UTIL_H

#include <CL/sycl.hpp>
#include <complex>

namespace syclct {

template<typename I, typename O> inline O bit_cast(I i) {
  return *reinterpret_cast<O*>(&i);
}

template <typename T> struct DataType { using T2 = T; };
template <typename T> struct DataType<cl::sycl::vec<T, 2>> { using T2 = std::complex<T>;};

// leading dim is col.
template<typename T>
inline void matrix_mem_copy(T *to_ptr, const T *from_ptr, int to_ld, int from_ld, int rows, int cols,
                            memcpy_direction direction) {
  using Ty = typename DataType<T>::T2;
  if(to_ptr==from_ptr && to_ld == from_ld){
    return;
  }
  if(to_ld == from_ld){
    sycl_memcpy((void*)to_ptr, (void*)from_ptr, sizeof(Ty)*to_ld*cols, direction);
  }else {
    auto to_ptr_t = to_ptr;
    auto from_ptr_t = from_ptr;
    to_ptr_t = to_ptr_t - to_ld;
    from_ptr_t = from_ptr_t - from_ld;
    for(int c = 0; c < cols; ++c) {
      to_ptr_t = to_ptr_t + to_ld;
      from_ptr_t = from_ptr_t + from_ld;
      sycl_memcpy((void*)(to_ptr_t), (void*)(from_ptr_t), sizeof(Ty)*rows, direction);
    }
  }
}

} // namespace syclct

#endif // SYCLCT_UTIL_H
