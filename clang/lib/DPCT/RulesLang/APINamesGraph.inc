//===--------------- APINamesGraph.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphInstantiate",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::instantiate",
                            ARG(0), ARG(1))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphInstantiate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphInstantiate"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph, DELETE_FACTORY_ENTRY("cudaGraphExecDestroy", ARG(0)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphExecDestroy",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphExecDestroy"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphLaunch", CALL(MapNames::getDpctNamespace() +
                                                   "experimental::launch",
                                               ARG(0), ARG(1))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphLaunch",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphLaunch"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphAddEmptyNode",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::add_empty_node",
                            ARG(0), ARG(1), ARG(2), ARG(3))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphAddEmptyNode",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphAddEmptyNode"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphAddDependencies",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::add_dependencies",
                            ARG(0), ARG(1), ARG(2), ARG(3))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphAddDependencies",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphAddDependencies"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphExecUpdate",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::update",
                            ARG(0), ARG(1), ARG(2))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphExecUpdate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphExecUpdate"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphNodeGetType",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::get_node_type",
                            ARG(0), ARG(1))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphNodeGetType",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphNodeGetType"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphGetNodes", CALL(MapNames::getDpctNamespace() +
                                                     "experimental::get_nodes",
                                                 ARG(0), ARG(1), ARG(2))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphGetNodes",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphGetNodes"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphGetRootNodes",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::get_root_nodes",
                            ARG(0), ARG(1), ARG(2))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphGetRootNodes",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphGetRootNodes"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph, DELETE_FACTORY_ENTRY("cudaGraphDestroy", ARG(0)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphDestroy",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphDestroy"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphAddKernelNode",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::add_kernel_node",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphAddKernelNode",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphAddKernelNode"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphKernelNodeGetParams",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::kernel_node_get_params",
                            ARG(0), ARG(1))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphKernelNodeGetParams",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphKernelNodeGetParams"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphKernelNodeSetParams",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::kernel_node_set_params",
                            ARG(0), ARG(1))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphKernelNodeSetParams",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphKernelNodeSetParams"),
                            ARG("--use-experimental-features=graph"))))
