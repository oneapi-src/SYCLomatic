//===----- graph.hpp ----------------------------*- C++ -*-----------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/queue.hpp"
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

static inline std::unordered_map<
    sycl::queue *,
    sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable> *>
    queue_graph_map;

static bool command_graph_begin_recording(sycl::queue *queue_ptr) {
  sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::modifiable>
      graph(queue_ptr->get_context(), queue_ptr->get_device());
  auto result = queue_graph_map.insert({queue_ptr, &graph});
  if (!result.second) {
    return false;
  }
  return graph->begin_recording(*queue_ptr);
}

static bool
command_graph_end_recording(sycl::queue *queue_ptr,
                            dpct::experimental::command_graph_ptr graph) {
  auto it = queue_graph_map.find(queue_ptr);
  if (it == queue_graph_map.end()) {
    return false;
  }
  graph = std::move(it->second);
  return graph->end_recording();
}

} // namespace experimental
} // namespace dpct
