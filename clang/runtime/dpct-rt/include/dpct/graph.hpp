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
#include <unordered_map>

namespace dpct {
namespace experimental {

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>
    *command_graph_ptr;

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>
    *command_graph_exec_ptr;

typedef sycl::ext::oneapi::experimental::node *node_ptr;

struct kernel_node_params {
  void *func{};
  dpct::dim3 grid_dim{};
  dpct::dim3 block_dim{};
  unsigned int shared_mem_bytes{};
  void **kernel_params{};

  std::vector<dpct::experimental::node_ptr> dependencies{};

public:
  void set_block_dim(const dpct::dim3 &block_dim) {
    this->block_dim = block_dim;
  }
  void set_grid_dim(const dpct::dim3 &grid_dim) { this->grid_dim = grid_dim; }
  void set_kernel_params(void **kernel_params) {
    this->kernel_params = kernel_params;
  }
  void set_func(void *func) { this->func = func; }
  void set_shared_mem_bytes(unsigned int shared_mem_bytes) {
    this->shared_mem_bytes = shared_mem_bytes;
  }
  dpct::dim3 get_block_dim() const { return block_dim; }
  dpct::dim3 get_grid_dim() const { return grid_dim; }
  void **get_kernel_params() const { return kernel_params; }
  void *get_func() const { return func; }
  unsigned int get_shared_mem_bytes() const { return shared_mem_bytes; }

  void add_dependency(dpct::experimental::node_ptr dependency) {
    dependencies.push_back(dependency);
  }
  const std::vector<dpct::experimental::node_ptr> &get_dependencies() const {
    return dependencies;
  }
  void update_dependency(const dpct::experimental::node_ptr &oldDependency,
                         const dpct::experimental::node_ptr &newDependency) {
    auto it =
        std::find(dependencies.begin(), dependencies.end(), oldDependency);
    if (it != dependencies.end()) {
      *it = newDependency;
    }
  }
};

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

  void begin_recording(sycl::queue *queue_ptr) {
    // Calling begin_recording on an already recording queue is a no-op in SYCL
    if (queue_graph_map.find(queue_ptr) != queue_graph_map.end()) {
      return;
    }
    auto graph = new sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable>(
        queue_ptr->get_context(), queue_ptr->get_device());
    auto result = queue_graph_map.insert({queue_ptr, graph});
    if (!result.second) {
      delete graph;
      return;
    }
    graph->begin_recording(*queue_ptr);
  }

  void end_recording(sycl::queue *queue_ptr,
                     dpct::experimental::command_graph_ptr *graph) {
    auto it = queue_graph_map.find(queue_ptr);
    if (it == queue_graph_map.end()) {
      return;
    }
    *graph = it->second;
    queue_graph_map.erase(it);
    (*graph)->end_recording();
  }

  void get_nodes(dpct::experimental::command_graph_ptr graph,
                 dpct::experimental::node_ptr *nodesArray,
                 std::size_t *numberOfNodes) {
    auto nodes = graph->get_nodes();
    nodes_map[graph] = nodes;
    *numberOfNodes = nodes.size();
    if (!nodesArray) {
      return;
    }
    for (std::size_t i = 0; i < *numberOfNodes; i++) {
      nodesArray[i] = &nodes_map[graph][i];
    }
  }

  void get_root_nodes(dpct::experimental::command_graph_ptr graph,
                      dpct::experimental::node_ptr *nodesArray,
                      std::size_t *numberOfNodes) {
    auto root_nodes = graph->get_root_nodes();
    root_nodes_map[graph] = root_nodes;
    *numberOfNodes = root_nodes.size();
    if (!nodesArray) {
      return;
    }
    for (std::size_t i = 0; i < *numberOfNodes; i++) {
      nodesArray[i] = &root_nodes_map[graph][i];
    }
  }

  void add_kernel_node(dpct::experimental::node_ptr *node,
                       dpct::experimental::command_graph_ptr graph,
                       dpct::experimental::node_ptr *dependencies,
                       std::size_t numberOfDependencies,
                       dpct::experimental::kernel_node_params *params) {
    node_graph_params_map[*node] = std::make_pair(graph, params);
    for (std::size_t i = 0; i < numberOfDependencies; i++) {
      params->add_dependency(dependencies[i]);
    }
    graph_kernel_node_params_map[graph].emplace_back(*node, params);
  }

  void launch(dpct::experimental::command_graph_exec_ptr execGraph,
              sycl::queue *queue) {
    // Retrieve the graph associated with execGraph
    auto graph = exec_graph_map[execGraph];
    auto &kernel_params_vector = graph_kernel_node_params_map[graph];
    for (std::size_t i = 0; i < kernel_params_vector.size(); i++) {
      auto &node_kernel_params_pair = kernel_params_vector[i];
      auto node_params = node_kernel_params_pair.second;
      const auto &dependency_ptrs = node_params->get_dependencies();
      std::vector<sycl::ext::oneapi::experimental::node> dependencies;
      dependencies.reserve(dependency_ptrs.size());
      for (const auto &dep_ptr : dependency_ptrs) {
        if (dep_ptr) {
          dependencies.push_back(*dep_ptr);
        }
      }
      auto new_node = new sycl::ext::oneapi::experimental::node(graph->add(
          [&](sycl::handler &cgh) {
            cgh.host_task([=]() {
              dpct::kernel_launcher::launch(
                  node_params->get_func(), node_params->get_grid_dim(),
                  node_params->get_block_dim(),
                  node_params->get_kernel_params(),
                  node_params->get_shared_mem_bytes(), queue);
            });
          },
          sycl::ext::oneapi::experimental::property::node::depends_on(
              dependencies)));
      if (i + 1 < kernel_params_vector.size()) {
        auto &next_node_params = kernel_params_vector[i + 1].second;
        auto next_dependency = next_node_params->get_dependencies()[i];
        next_node_params->update_dependency(next_dependency, new_node);
      }
      node_kernel_params_pair.first = new_node;
    }
    execGraph = new sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>(
        graph->finalize());
    queue->submit(
        [&](sycl::handler &cgh) { cgh.ext_oneapi_graph(*execGraph); });
  }

  void instantiate(dpct::experimental::command_graph_exec_ptr *execGraph,
                   dpct::experimental::command_graph_ptr graph) {
    exec_graph_map[*execGraph] = graph;
  }

  void kernel_node_get_params(dpct::experimental::node_ptr node,
                              dpct::experimental::kernel_node_params *params) {
    auto it = node_graph_params_map.find(node);
    if (it == node_graph_params_map.end()) {
      return;
    }
    *params = *(it->second.second);
  }

  void kernel_node_set_params(dpct::experimental::node_ptr node,
                              dpct::experimental::kernel_node_params *params) {
    node_graph_params_map[node].second = params;
  }

  void get_node_type(dpct::experimental::node_ptr node,
                     sycl::ext::oneapi::experimental::node_type *nodeType) {
    if (node_graph_params_map.find(node) != node_graph_params_map.end()) {
      *nodeType = sycl::ext::oneapi::experimental::node_type::kernel;
    } else {
      if (node) {
        *nodeType = node->get_type();
      } else {
        *nodeType = sycl::ext::oneapi::experimental::node_type::empty;
      }
    }
  }

private:
  std::unordered_map<sycl::queue *, command_graph_ptr> queue_graph_map;
  std::unordered_map<dpct::experimental::command_graph_ptr,
                     std::vector<sycl::ext::oneapi::experimental::node>>
      nodes_map;
  std::unordered_map<dpct::experimental::command_graph_ptr,
                     std::vector<sycl::ext::oneapi::experimental::node>>
      root_nodes_map;
  std::unordered_map<dpct::experimental::command_graph_exec_ptr,
                     dpct::experimental::command_graph_ptr>
      exec_graph_map;
  std::unordered_map<
      dpct::experimental::command_graph_ptr,
      std::vector<std::pair<dpct::experimental::node_ptr,
                            dpct::experimental::kernel_node_params *>>>
      graph_kernel_node_params_map;
  std::unordered_map<dpct::experimental::node_ptr,
                     std::pair<dpct::experimental::command_graph_ptr,
                               dpct::experimental::kernel_node_params *>>
      node_graph_params_map;
};
} // namespace detail

/// Begins recording commands into a command graph for a specific SYCL queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// will be recorded.
static inline void begin_recording(sycl::queue *queue_ptr) {
  detail::graph_mgr::instance().begin_recording(queue_ptr);
}

/// Ends the recording of commands into a command graph for a specific SYCL
/// queue.
/// \param [in] queue_ptr A pointer to the SYCL queue on which the commands
/// were recorded.
/// \param [out] graph A pointer to a command_graph_ptr pointer where the
/// command graph will be assigned.
static inline void end_recording(sycl::queue *queue_ptr,
                                 dpct::experimental::command_graph_ptr *graph) {
  detail::graph_mgr::instance().end_recording(queue_ptr, graph);
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

/// Gets the nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [out] nodesArray An array of node pointers where the
/// nodes will be assigned.
/// \param [out] numberOfNodes The number of nodes in the graph.
static void get_nodes(dpct::experimental::command_graph_ptr graph,
                      dpct::experimental::node_ptr *nodesArray,
                      std::size_t *numberOfNodes) {
  detail::graph_mgr::instance().get_nodes(graph, nodesArray, numberOfNodes);
}

/// Gets the root nodes in the command graph.
/// \param [in] graph A pointer to the command graph.
/// \param [out] nodesArray An array of node pointers where the
/// root nodes will be assigned.
/// \param [out] numberOfNodes The number of root nodes in the graph.
static void get_root_nodes(dpct::experimental::command_graph_ptr graph,
                           dpct::experimental::node_ptr *nodesArray,
                           std::size_t *numberOfNodes) {
  detail::graph_mgr::instance().get_root_nodes(graph, nodesArray,
                                               numberOfNodes);
}

static void add_kernel_node(dpct::experimental::node_ptr *node,
                            dpct::experimental::command_graph_ptr graph,
                            dpct::experimental::node_ptr *dependencies,
                            std::size_t numberOfDependencies,
                            dpct::experimental::kernel_node_params *params) {
  detail::graph_mgr::instance().add_kernel_node(node, graph, dependencies,
                                                numberOfDependencies, params);
}

static void instantiate(dpct::experimental::command_graph_exec_ptr *execGraph,
                        dpct::experimental::command_graph_ptr graph) {
  detail::graph_mgr::instance().instantiate(execGraph, graph);
}

static void launch(dpct::experimental::command_graph_exec_ptr execGraph,
                   sycl::queue *queue) {
  detail::graph_mgr::instance().launch(execGraph, queue);
}

static void
kernel_node_get_params(dpct::experimental::node_ptr node,
                       dpct::experimental::kernel_node_params *params) {
  detail::graph_mgr::instance().kernel_node_get_params(node, params);
}

static void
kernel_node_set_params(dpct::experimental::node_ptr node,
                       dpct::experimental::kernel_node_params *params) {
  detail::graph_mgr::instance().kernel_node_set_params(node, params);
}

static void
get_node_type(dpct::experimental::node_ptr node,
              sycl::ext::oneapi::experimental::node_type *nodeType) {
  detail::graph_mgr::instance().get_node_type(node, nodeType);
}

static void update(dpct::experimental::command_graph_exec_ptr graphExec,
                   dpct::experimental::command_graph_ptr graph,
                   int *updateResultInfo) {
  graphExec->update(*graph);
  if (!graphExec) {
    *updateResultInfo = 0;
  }
  *updateResultInfo = 1;
}

} // namespace experimental
} // namespace dpct
