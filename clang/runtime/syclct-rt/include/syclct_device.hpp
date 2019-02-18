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

//===--- syclct_device.hpp ------------------------------*- C++ -*---===//

#ifndef SYCLCT_DEVICE_H
#define SYCLCT_DEVICE_H

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

namespace syclct {

enum class compute_mode { default_, exclusive, prohibited, exclusive_process };

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
};

class sycl_device_info {
public:
  char *name() { return _name; }
  cl::sycl::id<3> &max_work_item_sizes() { return _max_work_item_sizes; }
  bool &host_unified_memory() { return _host_unified_memory; }
  int &major_version() { return _major; }
  int &minor_version() { return _minor; }
  int &get_integrated() { return _integrated; }
  int &max_clock_frequency() { return _frequency; }
  int &max_compute_units() { return _compute_units; }
  size_t &global_mem_size() { return _global_mem_size; }
  compute_mode &mode() { return _compute_mode; }
  // ...

private:
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  int _compute_units;
  size_t _global_mem_size;
  char _name[256];
  cl::sycl::id<3> _max_work_item_sizes;
  bool _host_unified_memory = false;
  compute_mode _compute_mode = compute_mode::default_;
};

class syclct_device : public cl::sycl::device {
public:
  syclct_device() : cl::sycl::device() {}
  syclct_device(const cl::sycl::device &base) : cl::sycl::device(base) {
    _default_queue = cl::sycl::queue(base, exception_handler);
  }

  int is_native_atomic_supported() { return 0; }
  //....

  int get_device_info(sycl_device_info &out) {
    sycl_device_info prop;
    std::strcpy(prop.name(), get_info<cl::sycl::info::device::name>().c_str());

    // Version string has the following format:
    // OpenCL<space><major.minor><space><vendor-specific-information>
    std::stringstream ver;
    ver << get_info<cl::sycl::info::device::version>();
    std::string item;
    std::getline(ver, item, ' '); // OpenCL
    std::getline(ver, item, '.'); // major
    prop.major_version() = std::stoi(item);
    std::getline(ver, item, ' '); // minor
    prop.minor_version() = std::stoi(item);

    prop.max_work_item_sizes() =
        get_info<cl::sycl::info::device::max_work_item_sizes>();
    prop.host_unified_memory() =
        get_info<cl::sycl::info::device::host_unified_memory>();
    prop.max_clock_frequency() =
        get_info<cl::sycl::info::device::max_clock_frequency>();
    prop.max_compute_units() =
        get_info<cl::sycl::info::device::max_compute_units>();
    prop.global_mem_size() =
        get_info<cl::sycl::info::device::global_mem_size>();
    //...
    out = prop;
    return SYCLCT_API_CALL_SUCCESS;
  }

  int reset() {
    // release ALL (TODO) resources and reset to initial state
    for (auto q : _queues) {
      // The destructor waits for all commands executing on the queue to
      // complete. It isn't possible to destroy a queue immediately. This is a
      // synchronization point in SYCL.
      q.~queue();
    }
    _queues.clear();
    return SYCLCT_API_CALL_SUCCESS;
  }

  cl::sycl::queue &default_queue() { return _default_queue; }

  void queues_wait_and_throw() {
    _default_queue.wait_and_throw();
    for (auto q : _queues) {
      q.wait_and_throw();
    }
  }

private:
  cl::sycl::queue _default_queue;
  std::set<cl::sycl::queue> _queues;
};

class device_manager {
public:
  device_manager() {
    std::vector<cl::sycl::device> sycl_gpu_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
    for (auto &dev : sycl_gpu_devs) {
      _devs.push_back(syclct_device(dev));
    }
    std::vector<cl::sycl::device> sycl_cpu_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::cpu);
    for (auto &dev : sycl_cpu_devs) {
      _devs.push_back(syclct_device(dev));
    }
  }
  syclct_device current_device() const {
    check_id(_current_device);
    return _devs[_current_device];
  }
  syclct_device get_device(unsigned int id) const {
    check_id(id);
    return _devs[id];
  }
  unsigned int current_device_id() const { return _current_device; }
  int select_device(unsigned int id) {
    check_id(id);
    _current_device = id;
    return SYCLCT_API_CALL_SUCCESS;
  }
  unsigned int device_count() { return _devs.size(); }

private:
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::string("invalid device id");
    }
  }
  std::vector<syclct_device> _devs;
  unsigned int _current_device = 0;
};

static device_manager &get_device_manager() {
  static device_manager d_m;
  return d_m;
}
static inline cl::sycl::queue &get_default_queue() {
  return syclct::get_device_manager().current_device().default_queue();
}

} // namespace syclct

#endif // SYCLCT_DEVICE_H
