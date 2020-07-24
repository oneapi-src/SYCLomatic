/******************************************************************************
*
* Copyright 2018 - 2020 Intel Corporation.
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

//===--- device.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>

namespace dpct {

/// DPC++ default exception handler
auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};

/// Device info
class device_info {
public:
  // get interface
  char *get_name() { return _name; }
  cl::sycl::id<3> get_max_work_item_sizes() { return _max_work_item_sizes; }
  bool get_host_unified_memory() { return _host_unified_memory; }
  int get_major_version() { return _major; }
  int get_minor_version() { return _minor; }
  int get_integrated() { return _integrated; }
  int get_max_clock_frequency() { return _frequency; }
  int get_max_compute_units() { return _max_compute_units; }
  int get_max_work_group_size() { return _max_work_group_size; }
  int get_max_sub_group_size() { return _max_sub_group_size; }
  int get_max_work_items_per_compute_unit() {
    return _max_work_items_per_compute_unit;
  }
  size_t *get_max_nd_range_size() { return _max_nd_range_size; }
  size_t get_global_mem_size() { return _global_mem_size; }
  size_t get_local_mem_size() { return _local_mem_size; }
  // set interface
  void set_name(const char *name) { std::strncpy(_name, name, 256); }
  void set_max_work_item_sizes(const cl::sycl::id<3> max_work_item_sizes) {
    _max_work_item_sizes = max_work_item_sizes;
  }
  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }
  void set_major_version(int major) { _major = major; }
  void set_minor_version(int minor) { _minor = minor; }
  void set_integrated(int integrated) { _integrated = integrated; }
  void set_max_clock_frequency(int frequency) { _frequency = frequency; }
  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }
  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }
  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }
  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }
  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }
  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++)
      _max_nd_range_size[i] = max_nd_range_size[i];
  }

private:
  char _name[256];
  cl::sycl::id<3> _max_work_item_sizes;
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
};

/// dpct device extension
class device_ext : public cl::sycl::device {
public:
  device_ext() : cl::sycl::device() {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto q : _queues) {
      delete q;
    }
  }
  device_ext(const cl::sycl::device &base) : cl::sycl::device(base) {
#ifdef DPCT_USM_LEVEL_NONE
    _default_queue = new cl::sycl::queue(base, exception_handler);
#else
    _default_queue = new cl::sycl::queue(base, exception_handler,
                                         cl::sycl::property::queue::in_order());
#endif
    _queues.insert(_default_queue);
    _saved_queue = _default_queue;
  }

  int is_native_atomic_supported() { return 0; }
  int get_major_version() {
    int major, minor;
    get_version(major, minor);
    return major;
  }

  int get_minor_version() {
    int major, minor;
    get_version(major, minor);
    return minor;
  }

  int get_max_compute_units() {
    return get_device_info().get_max_compute_units();
  }

  int get_max_clock_frequency() {
    return get_device_info().get_max_clock_frequency();
  }

  int get_integrated() { return get_device_info().get_integrated(); }

  void get_device_info(device_info &out) {
    device_info prop;
    prop.set_name(get_info<cl::sycl::info::device::name>().c_str());

    int major, minor;
    get_version(major, minor);
    prop.set_major_version(major);
    prop.set_minor_version(minor);

    prop.set_max_work_item_sizes(
        get_info<cl::sycl::info::device::max_work_item_sizes>());
    prop.set_host_unified_memory(
        get_info<cl::sycl::info::device::host_unified_memory>());

    // max_clock_frequency parameter is not supported on host device
    if (is_host()) {
      // This code may need to be updated. Currently max_clock_frequency for
      // host device is initialized with 1, in assumption that if other devices
      // exist and they are being selected based on this parameter, other
      // devices would have higher priority.
      prop.set_max_clock_frequency(1);
    } else {
      prop.set_max_clock_frequency(
          get_info<cl::sycl::info::device::max_clock_frequency>());
    }

    prop.set_max_compute_units(
        get_info<cl::sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(
        get_info<cl::sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(
        get_info<cl::sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<cl::sycl::info::device::local_mem_size>());

    // For Intel(R) oneAPI DPC++ Compiler, if current device does not support
    // "cl_intel_required_subgroup_size" extension, max_sub_group_size will be
    // initialized to one.
    // For other compilers, just initialize max_sub_group_size to one.
    // This code may need to be updated depending on subgroup support by other
    // compilers.
    size_t max_sub_group_size = 1;
#ifdef __SYCL_COMPILER_VERSION
    if (has_extension("cl_intel_required_subgroup_size")) {
      cl::sycl::vector_class<size_t> sub_group_sizes =
          get_info<cl::sycl::info::device::sub_group_sizes>();
      cl::sycl::vector_class<size_t>::const_iterator max_iter =
          std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
      max_sub_group_size = *max_iter;
    }
#endif
    prop.set_max_sub_group_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(
        get_info<cl::sycl::info::device::max_work_group_size>());
    int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    prop.set_max_nd_range_size(max_nd_range_size);

    out = prop;
  }

  device_info get_device_info() {
    device_info prop;
    get_device_info(prop);
    return prop;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto q : _queues) {
      // The destructor waits for all commands executing on the queue to
      // complete. It isn't possible to destroy a queue immediately. This is a
      // synchronization point in SYCL.
      delete q;
    }
    _queues.clear();
    // create new default queue.
#ifdef DPCT_USM_LEVEL_NONE
    _default_queue = new cl::sycl::queue(*this, exception_handler);
#else
    _default_queue = new cl::sycl::queue(*this, exception_handler,
                                         cl::sycl::property::queue::in_order());
#endif
    _queues.insert(_default_queue);
  }

  cl::sycl::queue &default_queue() { return *_default_queue; }

  void queues_wait_and_throw() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::set<cl::sycl::queue *> current_queues;
    std::copy(_queues.begin(), _queues.end(),
      std::inserter(current_queues, current_queues.begin()));
    lock.~lock_guard();
    for (auto q : current_queues) {
      q->wait_and_throw();
    }
  }
  cl::sycl::queue *create_queue(bool enable_exception_handler = false) {
    std::lock_guard<std::mutex> lock(m_mutex);
    cl::sycl::async_handler eh = {};
    if(enable_exception_handler) {
        eh = exception_handler;
    }
#ifdef DPCT_USM_LEVEL_NONE
    cl::sycl::queue *queue =
        new cl::sycl::queue(_default_queue->get_context(), *this, eh);
#else
    cl::sycl::queue *queue =
        new cl::sycl::queue(_default_queue->get_context(), *this, eh,
                            cl::sycl::property::queue::in_order());
#endif
    _queues.insert(queue);
    return queue;
  }
  void destroy_queue(cl::sycl::queue *&queue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _queues.erase(queue);
    delete queue;
    queue = NULL;
  }
  void set_saved_queue(cl::sycl::queue* q) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _saved_queue = q;
  }
  cl::sycl::queue* get_saved_queue() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return _saved_queue;
  }

private:
  void get_version(int &major, int &minor) {
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    std::string ver;
    ver = get_info<cl::sycl::info::device::version>();
    int i=0;
    while(i<ver.size()) {
      if(isdigit(ver[i]))
        break;
      i++;
    }
    major=std::stoi(&(ver[i]));
    while(i<ver.size()) {
      if(ver[i]=='.')
        break;
      i++;
    }
    i++;
    minor=std::stoi(&(ver[i]));
  }
  cl::sycl::queue *_default_queue;
  cl::sycl::queue *_saved_queue;
  std::set<cl::sycl::queue *> _queues;
  mutable std::mutex m_mutex;
};

/// device manager
class dev_mgr {
public:
  device_ext &current_device() {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(_current_device);
    return *_devs[_current_device];
  }
  device_ext &cpu_device() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (_cpu_device == -1) {
      throw std::string("no valid cpu device");
    } else {
      return *_devs[_cpu_device];
    }
  }
  device_ext &get_device(unsigned int id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return *_devs[id];
  }
  unsigned int current_device_id() const { return _current_device; }
  void select_device(unsigned int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _current_device = id;
  }
  unsigned int device_count() { return _devs.size(); }

  // Singleton to return the instance dev_mgr.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::mutex m_mutex;
  dev_mgr() {
    cl::sycl::device default_device =
        cl::sycl::device(cl::sycl::default_selector{});
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<cl::sycl::device> sycl_all_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::all);
    // Collect other devices except for the default device.
    const bool default_is_host = default_device.is_host();
    if (default_device.is_cpu())
      _cpu_device = 0;
    for (auto &dev : sycl_all_devs) {
      const bool dev_is_host = dev.is_host();
      if ((dev_is_host && default_is_host) ||
          (!dev_is_host && !default_is_host &&
           dev.get() == default_device.get())) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
  }
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::string("invalid device id");
    }
  }
  std::vector<std::shared_ptr<device_ext>> _devs;
  unsigned int _current_device = 0;
  int _cpu_device = -1;
};

/// Util function to get the defualt queue of current device in
/// dpct device manager.
static inline cl::sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}

/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}

/// Util function to get a device by id.
static inline device_ext &get_device(unsigned int id) {
  return dev_mgr::instance().get_device(id);
}

/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline cl::sycl::context get_default_context() {
  return dpct::get_default_queue().get_context();
}

/// Util function to get a cpu device.
static inline device_ext &cpu_device() {
  return dev_mgr::instance().cpu_device();
}

} // namespace dpct

#endif // __DPCT_DEVICE_HPP__
