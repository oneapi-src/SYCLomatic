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
    command_graph_ptr;

typedef sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>
    command_graph_exec_ptr;

} // namespace experimental
} // namespace dpct
