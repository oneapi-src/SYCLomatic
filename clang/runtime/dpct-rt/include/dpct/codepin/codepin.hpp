//==---- codepin.hpp -------------------------*- C++ -*---------------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_HPP__
#define __DPCT_CODEPIN_HPP__

#include "detail/json.hpp"
#include "detail/schema.hpp"
#include <memory>
#ifdef __NVCC__
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif
namespace dpct {
namespace experimental {

#ifdef __NVCC__
inline void synchronize(cudaStream_t stream) { cudaStreamSynchronize(stream); }

/// Generate API check point prolog.
/// \param api_name The UID of the function call.
/// \param stream The CUDA stream to synchronize the command execution.
/// \param args The schema string and variable value pair list.
template <class... Args>
void gen_prolog_API_CP(const std::string &api_name, cudaStream_t stream,
                       Args... args) {
  synchronize(stream);
  dpct::experimental::detail::gen_log_API_CP(api_name, args...);
}

/// Generate API check point epilog.
/// \param api_name The UID of the function call.
/// \param stream The CUDA stream to synchronize the command execution.
/// \param args The schema string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &api_name, cudaStream_t stream,
                       Args... args) {
  gen_prolog_API_CP(api_name, stream, args...);
}
#else
void synchronize(sycl::queue *q) { q->wait(); }

/// Generate API check point prolog.
/// \param api_name The UID of the function call.
/// \param queue The sycl queue to synchronize the command execution.
/// \param args The schema string and variable value pair list.
template <class... Args>
void gen_prolog_API_CP(const std::string &api_name, sycl::queue *queue,
                       Args... args) {
  synchronize(queue);
  dpct::experimental::detail::gen_log_API_CP(api_name, args...);
}

/// Generate API check point epilog.
/// \param api_name The UID of the function call.
/// \param stream The sycl queue to synchronize the command execution.
/// \param args The schema string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &api_name, sycl::queue *queue,
                       Args... args) {
  gen_prolog_API_CP(api_name, queue, args...);
}
#endif

inline std::map<void *, uint32_t> &getPointerSizeMap() {
  static std::map<void *, uint32_t> PtrSizeMap;
  return PtrSizeMap;
}

inline uint32_t getPointerSizeInBitsFromMap(void *ptr) {
  const std::map<void *, uint32_t> &PtrSizeMap = getPointerSizeMap();
  const auto &it = PtrSizeMap.find(ptr);
  return (it != PtrSizeMap.end()) ? it->second : 0;
}

} // namespace experimental
} // namespace dpct
#endif // End of __DPCT_CODEPIN_HPP__
