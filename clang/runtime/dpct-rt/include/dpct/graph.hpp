//===----- graph.hpp ----------------------------*- C++ -*-----------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace experimental {

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>
    *command_graph_ptr;

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>
    *command_graph_exec_ptr;

namespace detail {
class graph_mgr {
public:
  graph_mgr() = default;
  graph_mgr(const graph_mgr &) = delete;
  graph_mgr &operator=(const graph_mgr &) = delete;
  graph_mgr(graph_mgr &&) = delete;
  graph_mgr &operator=(graph_mgr &&) = delete;

  static graph_mgr &instance() {
    static graph_mgr instance;
    return instance;
  }

  bool begin_recording(sycl::queue *queue_ptr) {
    // Calling begin_recording on an already recording queue is a no-op in SYCL
    if (queue_graph_map.find(queue_ptr) != queue_graph_map.end()) {
      return false;
    }
    auto graph = new sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable>(
        queue_ptr->get_context(), queue_ptr->get_device());
    auto result = queue_graph_map.insert({queue_ptr, graph});
    if (!result.second) {
      delete graph;
      return false;
    }
    return graph->begin_recording(*queue_ptr);
  }

  bool end_recording(sycl::queue *queue_ptr,
                     dpct::experimental::command_graph_ptr *graph) {
    auto it = queue_graph_map.find(queue_ptr);
    if (it == queue_graph_map.end()) {
      return false;
    }
    *graph = it->second;
    queue_graph_map.erase(it);
    return (*graph)->end_recording();
  }

private:
  std::unordered_map<sycl::queue *, command_graph_ptr> queue_graph_map;
};
} // namespace detail

static inline bool command_graph_begin_recording(sycl::queue *queue_ptr) {
  return detail::graph_mgr::instance().begin_recording(queue_ptr);
}

static inline bool
command_graph_end_recording(sycl::queue *queue_ptr,
                            dpct::experimental::command_graph_ptr *graph) {
  return detail::graph_mgr::instance().end_recording(queue_ptr, graph);
}

} // namespace experimental
} // namespace dpct
