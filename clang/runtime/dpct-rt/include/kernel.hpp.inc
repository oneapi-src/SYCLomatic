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
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <cstdio>

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

class kernel_function {
public:
  kernel_function()                         : ptr{nullptr} {}
  kernel_function(dpct::kernel_functor ptr) : ptr{ptr}     {}

  operator void *() const { return ((void *)ptr); }

  void operator()(sycl::queue &q, const sycl::nd_range<3> &range,
                  unsigned int a, void **args, void **extra) {
    ptr(q,range,a,args,extra);
  }

private:
  dpct::kernel_functor ptr;
};

// DPCT_LABEL_BEGIN|get_kernel_function_info|dpct
// DPCT_DEPENDENCY_BEGIN
// Kernel|kernel_function_info
// Device|dev_mgr
// Device|dev_mgr_current_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const kernel_function &function) {
  kernel_info->max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
}
static kernel_function_info get_kernel_function_info(const kernel_function &function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      dpct::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
  return kernel_info;
}
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

namespace detail {

static std::vector<char> read_file_to_vector(const std::string &name) {
  std::ifstream ifs;
  ifs.open(name, std::ios::in | std::ios::binary);
  std::vector<char> content((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
  return content;
}

static void write_vector_to_file(const std::string &name, const std::vector<char> &blob) {
  auto file = std::fstream(name, std::ios::out | std::ios::binary);
  file.write(blob.data(), blob.size());
  file.close();
}

static uint16_t extract16(unsigned char const *const ptr) {
  uint16_t ret = 0;

  ret |= static_cast<uint16_t>(ptr[0]) <<  0;
  ret |= static_cast<uint16_t>(ptr[1]) <<  8;

  return(ret);
}

static uint32_t extract32(unsigned char const *const ptr) {
  uint32_t ret = 0;

  ret |= static_cast<uint32_t>(ptr[0]) <<  0;
  ret |= static_cast<uint32_t>(ptr[1]) <<  8;
  ret |= static_cast<uint32_t>(ptr[2]) << 16;
  ret |= static_cast<uint32_t>(ptr[3]) << 24;

  return(ret);
}

static uint64_t extract64(unsigned char const *const ptr) {
  uint64_t ret = 0;

  ret |= static_cast<uint64_t>(ptr[0]) <<  0;
  ret |= static_cast<uint64_t>(ptr[1]) <<  8;
  ret |= static_cast<uint64_t>(ptr[2]) << 16;
  ret |= static_cast<uint64_t>(ptr[3]) << 24;
  ret |= static_cast<uint64_t>(ptr[4]) << 32;
  ret |= static_cast<uint64_t>(ptr[5]) << 40;
  ret |= static_cast<uint64_t>(ptr[6]) << 48;
  ret |= static_cast<uint64_t>(ptr[7]) << 56;

  return(ret);
}

static uint64_t get_lib_size(char const *const blob) {
#ifdef _WIN32
  // See https://en.wikipedia.org/wiki/Portable_Executable
  // and https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
  // to understand the layout of a DLL header

  ///////////////////////////////////////////////////////////////////////
  unsigned char const *const ublob = reinterpret_cast<unsigned char const *const>(blob);
  // Analyze DOS stub
  if (ublob[0] != 0x4d ||
      ublob[1] != 0x5a) {
    throw std::runtime_error("Blob is not a Windows DLL.");
  }
  uint32_t pe_header_offset = extract32(ublob+0x3c);

  ///////////////////////////////////////////////////////////////////////
  // Ananlyze PE-header
  unsigned char const *const pe_header = ublob + pe_header_offset;

  // signature
  uint32_t pe_signature = extract32(pe_header+0);
  if (pe_signature!=0x00004550) {
    throw std::runtime_error("PE-header signature is not 0x00004550");
  }

  // machine
  uint16_t machine  = extract16(pe_header+4);
  if (machine!=0x8664) {
    throw std::runtime_error("Only DLLs for x64 supported");
  }

  // number of sections
  uint16_t number_of_sections  = extract16(pe_header+6);

  // sizeof optional header
  uint16_t sizeof_optional_header = extract16(pe_header+20);

  //magic
  uint16_t magic = extract16(pe_header+24);
  if (magic!=0x10b &&
      magic!=0x20b) {
    throw std::runtime_error("MAGIC is not 0x010b or 0x020b");
  }

  ///////////////////////////////////////////////////////////////////////
  // Analyze tail of optional header
  constexpr int coff_header_size = 24;

  unsigned char const *const tail_of_optional_header = pe_header + coff_header_size + sizeof_optional_header;
  if (extract64(tail_of_optional_header-8)!=0) {
    throw std::runtime_error("Optional header not zero-padded");
  }

  ///////////////////////////////////////////////////////////////////////
  // Analyze last section header
  constexpr int section_header_size = 40;
  unsigned char const *const last_section_header = tail_of_optional_header + section_header_size*(number_of_sections-1);

  uint32_t sizeof_raw_data     = extract32(last_section_header+16);
  uint32_t pointer_to_raw_data = extract32(last_section_header+20);

  return sizeof_raw_data + pointer_to_raw_data;
#else
  if (blob[0] != 0x7F || blob[1] != 'E' || blob[2] != 'L' || blob[3] != 'F')
    throw std::runtime_error("Blob is not in ELF format");

  if (blob[4] != 0x02)
    throw std::runtime_error("Only 64-bit headers are supported");

  if (blob[5] != 0x01)
    throw std::runtime_error("Only little-endian headers are supported");

  unsigned char const *const ublob = reinterpret_cast<unsigned char const *const>(blob);
  uint64_t e_shoff     = extract64(ublob+0x28);
  uint16_t e_shentsize = extract16(ublob+0x3A);
  uint16_t e_shnum     = extract16(ublob+0x3C);

  return e_shoff + (e_shentsize * e_shnum);
#endif
}

} // namespace detail

class module {
public:
  module()          : ptr{nullptr} {}
  module(void *ptr) : ptr{ptr}     {}

  operator void *() const { return ptr; }

void *get_ptr() {
  return(ptr);
}
private:
void *ptr;
};

namespace detail {

static std::string mkfilename()
{
  time_t rawtime;
  char buffer[80];

  time (&rawtime);
  sprintf(buffer,"%lld\n",(unsigned long long) rawtime);

  std::string str(buffer);
  std::hash<std::string> hasher;

  int i = hasher(str);

  sprintf(buffer,"%08X",i);

  return std::filesystem::temp_directory_path().string()
#ifndef _WIN32
         + "/"
#endif
         + "dpct"
         + buffer
#ifdef _WIN32
         + ".dll"
#endif
         ;
}

static module load_dl_from_vector(const std::vector<char> &blob) {
  const std::string name = mkfilename();
  write_vector_to_file(name, blob);
#ifdef _WIN32
  void *so = LoadLibraryA(name.c_str());
#else
  void *so = dlopen(name.c_str(), RTLD_LAZY);
#endif
  if (so == nullptr)
    throw std::runtime_error("Failed to load module");

  std::remove(name.c_str());

  return so;
}

} // namespace detail

static module load_sycl_lib(const std::string &name) {
  const auto blob = detail::read_file_to_vector(name);
  return detail::load_dl_from_vector(blob);
};

static module load_sycl_lib_mem(char const *const image) {
  const size_t size = detail::get_lib_size(image);

  std::vector<char> blob(size);
  std::memcpy(blob.data(), image, size);
  return detail::load_dl_from_vector(blob);
};

static void unload_sycl_lib(module &module) {
  dlclose(module.get_ptr());
};

static dpct::kernel_function get_kernel_function(module &module, const std::string &name) {
#ifdef _WIN32
  dpct::kernel_functor fn = (dpct::kernel_functor) GetProcAddress(static_cast<HMODULE>(module.get_ptr()), (name + std::string("_wrapper")).c_str());
#else
  dpct::kernel_functor fn = (dpct::kernel_functor) dlsym(module.get_ptr(), (name + std::string("_wrapper")).c_str());
#endif
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
