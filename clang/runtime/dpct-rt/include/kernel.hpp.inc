// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_KERNEL_HPP__
#define __DPCT_KERNEL_HPP__
// DPCT_COMMENT
// DPCT_COMMENT Example1:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_EMPTY
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT Example2:
// DPCT_COMMENT // DPCT_LABEL_BEGIN|FeatureNameDef|[Namespace]
// DPCT_COMMENT // DPCT_DEPENDENCY_BEGIN
// DPCT_COMMENT // FileID|FeatureNameRef
// DPCT_COMMENT [// FileID|FeatureNameRef]
// DPCT_COMMENT ...
// DPCT_COMMENT // DPCT_DEPENDENCY_END
// DPCT_COMMENT // DPCT_CODE
// DPCT_COMMENT some code
// DPCT_COMMENT // DPCT_LABEL_END
// DPCT_COMMENT
// DPCT_COMMENT For header file including dependency, please use predefined feature name:
// DPCT_COMMENT   local_include_dependency: dpct helper files
// DPCT_COMMENT   non_local_include_dependency: other header files

// DPCT_LABEL_BEGIN|non_local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
#include <sycl/sycl.hpp>
#include <dlfcn.h>
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END

namespace dpct {

// DPCT_LABEL_BEGIN|kernel_functor|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
typedef void (*kernel_functor)(sycl::queue &, const sycl::nd_range<3> &,
                               unsigned int, void **, void **);
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|kernel_function_info|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
struct kernel_function_info {
  int max_work_group_size = 0;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_kernel_function_info|dpct
// DPCT_DEPENDENCY_BEGIN
// Kernel|kernel_function_info
// Device|dev_mgr
// Device|dev_mgr_current_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
}
static kernel_function_info get_kernel_function_info(const void *function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
  return kernel_info;
}
// DPCT_LABEL_END

static std::vector<char> read_file_to_vector(const std::string &name) {
  std::ifstream ifs(name);
  std::vector<char> content((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
  return content;
}

static void write_vector_to_file(const std::string &name, const std::vector<char> &blob) {
  auto file = std::fstream(name, std::ios::out | std::ios::binary);
  file.write(blob.data(), blob.size());
  file.close();
}

static uint64_t get_size_from_elf(char const *const blob) {
  if (blob[0] != 0x7F || blob[1] != 'E' || blob[2] != 'L' || blob[3] != 'F')
    throw std::runtime_error("Blob is not in ELF format");

  if (blob[4] != 0x02)
    throw std::runtime_error("Only 64-bit headers are supported");

  if (blob[5] != 0x01)
    throw std::runtime_error("Only little-endian headers are supported");

  uint64_t e_shoff     = *(reinterpret_cast<const uint64_t *>(blob+0x28));
  uint16_t e_shentsize = *(reinterpret_cast<const uint16_t *>(blob+0x3A));
  uint16_t e_shnum     = *(reinterpret_cast<const uint16_t *>(blob+0x3C));

  return e_shoff + (e_shentsize * e_shnum);
}

class module {
public:
  module()          : ptr{nullptr} {}
  module(void *ptr) : ptr{ptr}     {}

void *get_ptr() {
  return(ptr);
}
private:
void *ptr;
};

static module load_dl_from_vector(const std::vector<char> &blob) {
  const std::string name = std::tmpnam(nullptr);
  write_vector_to_file(name, blob);
  void *so = dlopen(name.c_str(), RTLD_LAZY);
  if (so == nullptr)
    throw std::runtime_error("Failed to load module");
  return so;
}

static module load_sycl_lib(const std::string &name) {
  const auto blob = read_file_to_vector(name);
  return load_dl_from_vector(blob);
};

static module load_sycl_lib_mem(char const *const image) {
  const size_t size = get_size_from_elf(image);

  std::vector<char> blob(size);
  std::memcpy(blob.data(), image, size);
  return load_dl_from_vector(blob);
};

  class kernel_function {
  public:
    kernel_function()                         : ptr{nullptr} {}
    kernel_function(dpct::kernel_functor ptr) : ptr{ptr}     {}

    void kernel_function::operator()(sycl::queue &q, const sycl::nd_range<3> &range,
                                     unsigned int a, void **args, void **extra) {
      ptr(q,range,a,args,extra);
    }

  private:
    dpct::kernel_functor ptr;
  };

static dpct::kernel_function get_kernel_function(module &module, const std::string &name) {
  dpct::kernel_functor fn = (dpct::kernel_functor) dlsym(module.get_ptr(), (name + std::string("_wrapper")).c_str());
  if (fn == nullptr)
    throw std::runtime_error("Failed to get function");
  return fn;
}

static void invoke_kernel_function(dpct::kernel_function &function, sycl::queue &queue,
                                   sycl::range<3> a,
                                   sycl::range<3> b,
                                   unsigned int localMemSize, void **kernelParams, void **extra) {
  function(queue,
           sycl::nd_range<3>(a*b,b),
           localMemSize,kernelParams,extra);
}

} // namespace dpct
#endif // __DPCT_KERNEL_HPP__
