// DPCT_LABEL_BEGIN|License|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
//==---- device.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// DPCT_LABEL_END

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__
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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <map>
#include <vector>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|local_include_dependency|
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
// DPCT_LABEL_END


namespace dpct {

// DPCT_LABEL_BEGIN|exception_handler|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
/// SYCL default exception handler
auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|typedef_event_ptr|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
typedef sycl::event *event_ptr;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|typedef_queue_ptr|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
typedef sycl::queue *queue_ptr;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|destroy_event|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|typedef_event_ptr
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Destroy \p event pointed memory.
///
/// \param event Pointer to the sycl::event address.
static void destroy_event(event_ptr event) {
    delete event;
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_info|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|device_info_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
class device_info {
public:
// DPCT_LABEL_END
  // get interface
// DPCT_LABEL_BEGIN|device_info_get_name|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  const char *get_name() const { return _name; }
  char *get_name() { return _name; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_work_item_sizes|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  template <typename WorkItemSizesTy = sycl::id<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() const {
    if constexpr (std::is_same_v<WorkItemSizesTy, sycl::id<3>>)
      return _max_work_item_sizes;
    else
      return _max_work_item_sizes_i;
  }
  template <typename WorkItemSizesTy = sycl::id<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::id<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() {
    if constexpr (std::is_same_v<WorkItemSizesTy, sycl::id<3>>)
      return _max_work_item_sizes;
    else
      return _max_work_item_sizes_i;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_host_unified_memory|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  bool get_host_unified_memory() const { return _host_unified_memory; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_major_version|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_major_version() const { return _major; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_minor_version|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_minor_version() const { return _minor; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_integrated|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_integrated() const { return _integrated; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_clock_frequency|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_clock_frequency() const { return _frequency; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_compute_units|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_compute_units() const { return _max_compute_units; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_work_group_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_work_group_size() const { return _max_work_group_size; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_sub_group_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_sub_group_size() const { return _max_sub_group_size; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_work_items_per_compute_unit|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_work_items_per_compute_unit() const {
    return _max_work_items_per_compute_unit;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_max_nd_range_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size() const {
    if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
      return _max_nd_range_size;
    else
      return _max_nd_range_size_i;
  }
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size() {
    if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
      return _max_nd_range_size;
    else
      return _max_nd_range_size_i;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_global_mem_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_global_mem_size() const { return _global_mem_size; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_local_mem_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  size_t get_local_mem_size() const { return _local_mem_size; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_memory_clock_rate|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_memory_clock_rate() const { return _memory_clock_rate; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_get_memory_bus_width|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_memory_bus_width() const { return _memory_bus_width; }
// DPCT_LABEL_END
  // set interface
// DPCT_LABEL_BEGIN|device_info_set_name|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_name(const char* name) {
    size_t length = strlen(name);
    if (length < 256) {
      std::memcpy(_name, name, length + 1);
    } else {
      std::memcpy(_name, name, 255);
      _name[255] = '\0';
    }
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_work_item_sizes|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes) {
    _max_work_item_sizes = max_work_item_sizes;
    for (int i = 0; i < 3; ++i)
      _max_work_item_sizes_i[i] = max_work_item_sizes[i];
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_host_unified_memory|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_major_version|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_major_version(int major) { _major = major; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_minor_version|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_minor_version(int minor) { _minor = minor; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_integrated|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_integrated(int integrated) { _integrated = integrated; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_clock_frequency|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_clock_frequency(int frequency) { _frequency = frequency; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_compute_units|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_global_mem_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_local_mem_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_work_group_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_sub_group_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_work_items_per_compute_unit|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_max_nd_range_size|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++) {
      _max_nd_range_size[i] = max_nd_range_size[i];
      _max_nd_range_size_i[i] = max_nd_range_size[i];
    }
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_memory_clock_rate|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_memory_clock_rate(int memory_clock_rate){
    _memory_clock_rate = memory_clock_rate;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_info_set_memory_bus_width|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_memory_bus_width(int memory_bus_width){
    _memory_bus_width = memory_bus_width;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_info_1|dpct
// DPCT_PARENT_FEATURE|device_info
// DPCT_DEPENDENCY_BEGIN
// Device|device_info
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  char _name[256];
  sycl::id<3> _max_work_item_sizes;
  int _max_work_item_sizes_i[3];
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  int _memory_clock_rate;
  int _memory_bus_width;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
  int _max_nd_range_size_i[3];
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|exception_handler
// Device|device_ext_2
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// dpct device extension
class device_ext : public sycl::device {
public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &task : _tasks) {
      if (task.joinable())
        task.join();
    }
    _tasks.clear();
    _queues.clear();
  }
  device_ext(const sycl::device &base)
      : sycl::device(base), _ctx(*this) {
#ifdef DPCT_USM_LEVEL_NONE
    _queues.push_back(
        std::make_shared<sycl::queue>(_ctx, base, exception_handler));
#else
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, base, exception_handler, sycl::property::queue::in_order()));
#endif
    _saved_queue = _default_queue = _queues[0].get();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_is_native_atomic_supported|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int is_native_atomic_supported() { return 0; }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_get_major_version|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_version
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_major_version() const {
    int major, minor;
    get_version(major, minor);
    return major;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_minor_version|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_version
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_minor_version() const {
    int major, minor;
    get_version(major, minor);
    return minor;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_max_compute_units|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_device_info_return_info
// Device|device_info_get_max_compute_units
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_compute_units() const {
    return get_device_info().get_max_compute_units();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_max_clock_frequency|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_device_info_return_info
// Device|device_info_get_max_clock_frequency
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_max_clock_frequency() const {
    return get_device_info().get_max_clock_frequency();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_integrated|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_device_info_return_info
// Device|device_info_get_integrated
// DPCT_DEPENDENCY_END
// DPCT_CODE
  int get_integrated() const { return get_device_info().get_integrated(); }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_memory_info|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_device_info_return_info
// Device|device_info_get_global_mem_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Get the number of bytes of free and total memory on the SYCL device.
  /// \param [out] free_memory The number of bytes of free memory on the SYCL device.
  /// \param [out] total_memory The number of bytes of total memory on the SYCL device.
  void get_memory_info(size_t &free_memory, size_t &total_memory) {
#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    if (!has(sycl::aspect::ext_intel_free_memory)) {
      std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
      free_memory = 0;
    } else {
      free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
    }
#else
    std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
    free_memory = 0;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#else
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
    total_memory = get_device_info().get_global_mem_size();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_device_info_return_void|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_info
// Device|device_ext_get_version
// Device|device_info_set_name
// Device|device_info_set_major_version
// Device|device_info_set_minor_version
// Device|device_info_set_max_work_item_sizes
// Device|device_info_set_host_unified_memory
// Device|device_info_set_max_clock_frequency
// Device|device_info_set_max_compute_units
// Device|device_info_set_max_work_group_size
// Device|device_info_set_global_mem_size
// Device|device_info_set_local_mem_size
// Device|device_info_set_max_sub_group_size
// Device|device_info_set_max_work_items_per_compute_unit
// Device|device_info_set_max_nd_range_size
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void get_device_info(device_info &out) const {
    device_info prop;
    prop.set_name(get_info<sycl::info::device::name>().c_str());

    int major, minor;
    get_version(major, minor);
    prop.set_major_version(major);
    prop.set_minor_version(minor);

    prop.set_max_work_item_sizes(
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION<20220902)
        // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes is an enum class element
        get_info<sycl::info::device::max_work_item_sizes>());
#else
        // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by an int
        get_info<sycl::info::device::max_work_item_sizes<3>>());
#endif
    prop.set_host_unified_memory(
        this->has(sycl::aspect::usm_host_allocations));

    prop.set_max_clock_frequency(
        get_info<sycl::info::device::max_clock_frequency>());

    prop.set_max_compute_units(
        get_info<sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(
        get_info<sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(
        get_info<sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<sycl::info::device::local_mem_size>());
    if (this->has(aspect::ext_intel_memory_clock_rate))
    {
      prop.set_memory_clock_rate(
          this->get_info<ext::intel::info::device::memory_clock_rate>());
    }
    else
    {
      prop.set_memory_clock_rate(0);
    }
    if (this->has(aspect::ext_intel_memory_bus_width))
    {
      prop.set_memory_bus_width(
          this->get_info<ext::intel::info::device::memory_bus_width>());
    }
    else
    {
      prop.set_memory_bus_width(0);
    }

    size_t max_sub_group_size = 1;
    std::vector<size_t> sub_group_sizes =
        get_info<sycl::info::device::sub_group_sizes>();

    for (const auto &sub_group_size : sub_group_sizes) {
      if (max_sub_group_size < sub_group_size)
        max_sub_group_size = sub_group_size;
    }

    prop.set_max_sub_group_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(
        get_info<sycl::info::device::max_work_group_size>());
    int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    prop.set_max_nd_range_size(max_nd_range_size);

    out = prop;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_get_device_info_return_info|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_get_device_info_return_void
// DPCT_DEPENDENCY_END
// DPCT_CODE
  device_info get_device_info() const {
    device_info prop;
    get_device_info(prop);
    return prop;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_reset|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    // The queues are shared_ptrs and the ref counts of the shared_ptrs increase
    // only in wait_and_throw(). If there is no other thread calling
    // wait_and_throw(), the queues will be destructed. The destructor waits for
    // all commands executing on the queue to complete. It isn't possible to
    // destroy a queue immediately. This is a synchronization point in SYCL.
    _queues.clear();
    // create new default queue.
#ifdef DPCT_USM_LEVEL_NONE
    _queues.push_back(
        std::make_shared<sycl::queue>(_ctx, *this, exception_handler));
#else
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, exception_handler, sycl::property::queue::in_order()));
#endif
    _saved_queue = _default_queue = _queues.front().get();
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_default_queue|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sycl::queue &default_queue() { return *_default_queue; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_queues_wait_and_throw|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void queues_wait_and_throw() {
    std::unique_lock<std::mutex> lock(m_mutex);
    std::vector<std::shared_ptr<sycl::queue>> current_queues(
        _queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      q->wait_and_throw();
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_create_queue|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sycl::queue *create_queue(bool enable_exception_handler = false) {
    std::lock_guard<std::mutex> lock(m_mutex);
    sycl::async_handler eh = {};
    if (enable_exception_handler) {
      eh = exception_handler;
    }
#ifdef DPCT_USM_LEVEL_NONE
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh));
#else
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh,
        sycl::property::queue::in_order()));
#endif
    return _queues.back().get();
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_destroy_queue|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                  [=](const std::shared_ptr<sycl::queue> &q) -> bool {
                                    return q.get() == queue;
                                  }),
                   _queues.end());
    queue = nullptr;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_set_saved_queue|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void set_saved_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _saved_queue = q;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_get_saved_queue|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sycl::queue* get_saved_queue() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return _saved_queue;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_get_context|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sycl::context get_context() const { return _ctx; }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|device_ext_1|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_get_version|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void get_version(int &major, int &minor) const {
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    std::string ver;
    ver = get_info<sycl::info::device::version>();
    std::string::size_type i = 0;
    while (i < ver.size()) {
      if (isdigit(ver[i]))
        break;
      i++;
    }
    major = std::stoi(&(ver[i]));
    while (i < ver.size()) {
      if (ver[i] == '.')
        break;
      i++;
    }
    i++;
    minor = std::stoi(&(ver[i]));
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_add_task|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
  void add_task(std::thread &&task) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _tasks.push_back(std::move(task));
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_async_dpct_free_friend_decl|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
  friend void async_dpct_free(std::vector<void *>,
                              std::vector<sycl::event>,
                              sycl::queue &);
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|device_ext_2|dpct
// DPCT_PARENT_FEATURE|device_ext
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|device_ext_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  sycl::queue *_default_queue;
  sycl::queue *_saved_queue;
  sycl::context _ctx;
  std::vector<std::shared_ptr<sycl::queue>> _queues;
  mutable std::mutex m_mutex;
  std::vector<std::thread> _tasks;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_tid|dpct
// DPCT_DEPENDENCY_EMPTY
// DPCT_CODE
static inline unsigned int get_tid() {
#if defined(__linux__)
  return syscall(SYS_gettid);
#elif defined(_WIN64)
  return GetCurrentThreadId();
#else
#error "Only support Windows and Linux."
#endif
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dev_mgr|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|device_ext
// Device|dev_mgr_2
// Device|dev_mgr_3
// Device|dev_mgr_4
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// device manager
class dev_mgr {
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_1|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr
// DPCT_DEPENDENCY_END
// DPCT_CODE
public:
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_current_device|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_check_id
// Device|dev_mgr_1
// Device|dev_mgr_current_device_id
// DPCT_DEPENDENCY_END
// DPCT_CODE
  device_ext &current_device() {
    unsigned int dev_id=current_device_id();
    check_id(dev_id);
    return *_devs[dev_id];
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_cpu_device|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  device_ext &cpu_device() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (_cpu_device == -1) {
      throw std::runtime_error("no valid cpu device");
    } else {
      return *_devs[_cpu_device];
    }
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_get_device|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_check_id
// Device|dev_mgr_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  device_ext &get_device(unsigned int id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return *_devs[id];
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_current_device_id|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_1
// Device|get_tid
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned int current_device_id() const {
   std::lock_guard<std::mutex> lock(m_mutex);
   auto it=_thread2dev_map.find(get_tid());
   if(it != _thread2dev_map.end())
      return it->second;
    return DEFAULT_DEVICE_ID;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_select_device|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_1
// Device|dev_mgr_check_id
// Device|get_tid
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void select_device(unsigned int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()]=id;
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_device_count|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned int device_count() { return _devs.size(); }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dev_mgr_get_device_id|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_1
// DPCT_DEPENDENCY_END
// DPCT_CODE
  unsigned int get_device_id(const sycl::device &dev) {
    unsigned int id = 0;
    for(auto dev_item : _devs) {
      if (*dev_item == dev) {
        break;
      }
      id++;
    }
    return id;
  }
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dev_mgr_2|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr
// Device|dev_mgr_3
// Device|dev_mgr_4
// DPCT_DEPENDENCY_END
// DPCT_CODE
  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|dev_mgr_3|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr
// Device|dev_mgr_2
// Device|dev_mgr_4
// DPCT_DEPENDENCY_END
// DPCT_CODE
private:
  mutable std::mutex m_mutex;
  dev_mgr() {
    sycl::device default_device =
        sycl::device(sycl::default_selector_v);
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<sycl::device> sycl_all_devs =
        sycl::device::get_devices(sycl::info::device_type::all);
    // Collect other devices except for the default device.
    if (default_device.is_cpu())
      _cpu_device = 0;
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_check_id|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr
// DPCT_DEPENDENCY_END
// DPCT_CODE
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }
// DPCT_LABEL_END
// DPCT_LABEL_BEGIN|dev_mgr_4|dpct
// DPCT_PARENT_FEATURE|dev_mgr
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr
// Device|dev_mgr_2
// Device|dev_mgr_3
// DPCT_DEPENDENCY_END
// DPCT_CODE
  std::vector<std::shared_ptr<device_ext>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<unsigned int, unsigned int> _thread2dev_map;
  int _cpu_device = -1;
};
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_default_queue|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_current_device
// Device|device_ext_default_queue
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get the default queue of current device in
/// dpct device manager.
static inline sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_current_device_id|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_current_device_id
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get the id of current device in
/// dpct device manager.
static inline unsigned int get_current_device_id() {
  return dev_mgr::instance().current_device_id();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_current_device|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_current_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_device|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_get_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get a device by id.
static inline device_ext &get_device(unsigned int id) {
  return dev_mgr::instance().get_device(id);
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|get_default_context|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|get_current_device
// Device|device_ext_get_context
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline sycl::context get_default_context() {
  return dpct::get_current_device().get_context();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|cpu_device|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_cpu_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
/// Util function to get a CPU device.
static inline device_ext &cpu_device() {
  return dev_mgr::instance().cpu_device();
}
// DPCT_LABEL_END

// DPCT_LABEL_BEGIN|select_device|dpct
// DPCT_DEPENDENCY_BEGIN
// Device|dev_mgr_select_device
// DPCT_DEPENDENCY_END
// DPCT_CODE
static inline unsigned int select_device(unsigned int id){
  dev_mgr::instance().select_device(id);
  return id;
}
// DPCT_LABEL_END

} // namespace dpct

#endif // __DPCT_DEVICE_HPP__
