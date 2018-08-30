//===--- syclct_device.hpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_DEVICE_H
#define SYCLCT_DEVICE_H

#include <CL/sycl.hpp>
#include <iostream>
#include <set>

namespace syclct {

enum class compute_mode { default_, exclusive, prohibited, exclusive_process };

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    }
    catch (cl::sycl::exception const& e) {
      std::cerr << "Caught asynchronous SYCL exception:\n" << e.what()
        << std::endl;
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
  int &max_clock_frequency() { return _frequency; }
  int &max_compute_units() { return _compute_units; }
  compute_mode &mode() { return _compute_mode; }
  // ...

private:
  int _major;
  int _minor;
  int _frequency;
  int _compute_units;
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

  sycl_device_info get_device_info() {
    sycl_device_info prop;
    std::strcpy(prop.name(), get_info<cl::sycl::info::device::name>().c_str());
    std::string ver = get_info<cl::sycl::info::device::version>();
    std::size_t found = ver.find('.');
    if (found == std::string::npos) {
      prop.major_version() = 0;
      prop.minor_version() = 0;
    } else {
      // FIXME: Example of ver string that I have: "OpenCL 1.2 (Build 25)"
      // And it obviousely crashes with code below.
//      prop.major_version() = std::stoi(ver.substr(0, found - 1));
//      prop.minor_version() = std::stoi(ver.substr(0, found + 1));
    }
    prop.max_work_item_sizes() =
        get_info<cl::sycl::info::device::max_work_item_sizes>();
    prop.host_unified_memory() =
        get_info<cl::sycl::info::device::host_unified_memory>();
    prop.max_clock_frequency() =
        get_info<cl::sycl::info::device::max_clock_frequency>();
    prop.max_compute_units() =
        get_info<cl::sycl::info::device::max_compute_units>();
    //...
    return prop;
  }

  int reset() {
    // release ALL (TODO) resources and reset to initial state
    for (auto q : _queues) {
      // The destructor waits for all commands executing on the queue to complete.
      // It isn't possible to destroy a queue immediately.
      // This is a synchronization point in SYCL.
      q.~queue();
    }
    _queues.clear();
    return 0;
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
    std::vector<cl::sycl::device> sycl_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
    for (auto& dev : sycl_devs) {
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
  syclct_device select_device(unsigned int id) {
    check_id(id);
    _current_device = id;
    return _devs[_current_device];
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

} // namespace syclct

#endif // SYCLCT_DEVICE_H
