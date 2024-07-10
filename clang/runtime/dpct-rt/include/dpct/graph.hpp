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

typedef sycl::ext::oneapi::experimental::node *node_ptr;

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

#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
  bool begin_recording(sycl::queue *queue_ptr) {
#else
  void begin_recording(sycl::queue *queue_ptr) {
#endif
    // Calling begin_recording on an already recording queue is a no-op in SYCL
    if (queue_graph_map.find(queue_ptr) != queue_graph_map.end()) {
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
      return false;
#else
      return;
#endif
    }
    auto graph = new sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable>(
        queue_ptr->get_context(), queue_ptr->get_device());
    auto result = queue_graph_map.insert({queue_ptr, graph});
    if (!result.second) {
      delete graph;
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
      return false;
#else
      return;
#endif
    }
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
    return graph->begin_recording(*queue_ptr);
#else
    graph->begin_recording(*queue_ptr);
    return;
#endif
  }

#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
  bool end_recording(sycl::queue *queue_ptr,
#else
  void end_recording(sycl::queue *queue_ptr,
#endif
                     dpct::experimental::command_graph_ptr *graph) {
    auto it = queue_graph_map.find(queue_ptr);
    if (it == queue_graph_map.end()) {
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
      return false;
#else
      return;
#endif
    }
    *graph = it->second;
    queue_graph_map.erase(it);
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
    return;
#else
    (*graph)->end_recording();
    return;
#endif
  }

private:
  std::unordered_map<sycl::queue *, command_graph_ptr> queue_graph_map;
};
} // namespace detail

/// Begins recording commands into a command graph for a specific SYCL queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// will be recorded.
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
/// \returns `true` if the recording is successfully started.
static inline bool begin_recording(sycl::queue *queue_ptr) {
  return detail::graph_mgr::instance().begin_recording(queue_ptr);
#else
static inline void begin_recording(sycl::queue *queue_ptr) {
  detail::graph_mgr::instance().begin_recording(queue_ptr);
#endif
}

/// Ends the recording of commands into a command graph for a specific SYCL
/// queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// were recorded.
/// \param [out] graph A pointer to a command_graph_ptr pointer where the
/// command graph will be assigned.
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20240427)
/// \returns `true` if the recording is successfully ended and the
/// graph is assigned.
static inline bool end_recording(sycl::queue *queue_ptr,
                                 dpct::experimental::command_graph_ptr *graph) {
  return detail::graph_mgr::instance().end_recording(queue_ptr, graph);
#else
static inline void end_recording(sycl::queue *queue_ptr,
                                 dpct::experimental::command_graph_ptr *graph) {
  detail::graph_mgr::instance().end_recording(queue_ptr, graph);
#endif
}


/// Adds an empty node to the command graph with optional
/// dependencies.
/// \param [out] newNode A pointer to the node_ptr that will be
/// added to the graph.
/// \param [in] graph A pointer to the command graph.
/// \param [in] dependenciesArray An array of node pointers
/// representing the dependencies of the new node.
/// \param [in] numberOfDependencies The number of dependencies in
/// the dependenciesArray.
static void
add_empty_node(dpct::experimental::node_ptr *newNode,
               dpct::experimental::command_graph_ptr graph,
               const dpct::experimental::node_ptr *dependenciesArray,
               std::size_t numberOfDependencies) {
  if (numberOfDependencies == 0) {
    *newNode = new sycl::ext::oneapi::experimental::node(graph->add());
    return;
  }
  std::vector<sycl::ext::oneapi::experimental::node> dependencies;
  for (std::size_t i = 0; i < numberOfDependencies; i++) {
    dependencies.push_back(*dependenciesArray[i]);
  }
  *newNode =
      new sycl::ext::oneapi::experimental::node(graph->add(sycl::property_list{
          sycl::ext::oneapi::experimental::property::node::depends_on(
              dependencies)}));
}

/// Adds dependencies between nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [in] fromNodes An array of node pointers representing
/// the source nodes.
/// \param [in] toNodes An array of node pointers representing
/// the destination nodes.
/// \param [in] numberOfDependencies The number of dependencies
/// to be added.
static void add_dependencies(dpct::experimental::command_graph_ptr graph,
                             const dpct::experimental::node_ptr *fromNodes,
                             const dpct::experimental::node_ptr *toNodes,
                             std::size_t numberOfDependencies) {
  for (std::size_t i = 0; i < numberOfDependencies; i++) {
    graph->make_edge(*fromNodes[i], *toNodes[i]);
  }
}

} // namespace experimental
} // namespace dpct
