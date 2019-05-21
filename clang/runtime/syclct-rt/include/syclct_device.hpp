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
  //get interface
  char *get_name() { return _name; }
  cl::sycl::id<3> get_max_work_item_sizes() { return _max_work_item_sizes; }
  bool get_host_unified_memory() { return _host_unified_memory; }
  int get_major_version() { return _major; }
  int get_minor_version() { return _minor; }
  int get_integrated() { return _integrated; }
  int get_max_clock_frequency() { return _frequency; }
  int get_max_compute_units() { return _compute_units; }
  size_t get_global_mem_size() { return _global_mem_size; }
  compute_mode get_mode() { return _compute_mode; }
  // set interface
  void set_name(const char* name) {std::strncpy(_name, name,256);}
  void set_max_work_item_sizes(const cl::sycl::id<3> max_work_item_sizes) {_max_work_item_sizes=max_work_item_sizes;}
  void set_host_unified_memory(bool host_unified_memory) {_host_unified_memory=host_unified_memory;}
  void set_major_version(int major) {_major=major;}
  void set_minor_version(int minor) {_minor=minor;}
  void set_integrated(int integrated) {_integrated=integrated;}
  void set_max_clock_frequency(int frequency) {_frequency=frequency;}
  void set_max_compute_units(int compute_units) {_compute_units=compute_units;}
  void set_global_mem_size(size_t global_mem_size) {_global_mem_size=global_mem_size;}
  void set_mode(compute_mode compute_mode){_compute_mode=compute_mode;}

private:
  char _name[256];
  cl::sycl::id<3> _max_work_item_sizes;
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  int _compute_units;
  size_t _global_mem_size;
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

  void get_device_info(sycl_device_info &out) {
    sycl_device_info prop;
    prop.set_name(get_info<cl::sycl::info::device::name>().c_str());

    // Version string has the following format:
    // OpenCL<space><major.minor><space><vendor-specific-information>
    std::stringstream ver;
    ver << get_info<cl::sycl::info::device::version>();
    std::string item;
    std::getline(ver, item, ' '); // OpenCL
    std::getline(ver, item, '.'); // major
    prop.set_major_version(std::stoi(item));
    std::getline(ver, item, ' '); // minor
    prop.set_minor_version(std::stoi(item));

    prop.set_max_work_item_sizes(get_info<cl::sycl::info::device::max_work_item_sizes>());
    prop.set_host_unified_memory(get_info<cl::sycl::info::device::host_unified_memory>());
    prop.set_max_clock_frequency(get_info<cl::sycl::info::device::max_clock_frequency>());
    prop.set_max_compute_units(get_info<cl::sycl::info::device::max_compute_units>());
    prop.set_global_mem_size(get_info<cl::sycl::info::device::global_mem_size>());
    //...
    out = prop;
  }

  void reset() {
    // release ALL (TODO) resources and reset to initial state
    for (auto q : _queues) {
      // The destructor waits for all commands executing on the queue to
      // complete. It isn't possible to destroy a queue immediately. This is a
      // synchronization point in SYCL.
      q.~queue();
    }
    _queues.clear();
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
  void select_device(unsigned int id) {
    check_id(id);
    _current_device = id;
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
