//===--------------- APINamesGraph.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaGraphInstantiate", DEREF(0),
        NEW(MapNames::getClNamespace() +
                "ext::oneapi::experimental::command_graph<" +
                MapNames::getClNamespace() +
                "ext::oneapi::experimental::graph_state::executable>",
            MEMBER_CALL(ARG(1), true, "finalize")))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphInstantiate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphInstantiate"),
                            ARG("--use-experimental-features=graph")))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph, DELETE_FACTORY_ENTRY("cudaGraphExecDestroy", ARG(0)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphExecDestroy",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphExecDestroy"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    MEMBER_CALL_FACTORY_ENTRY("cudaGraphLaunch", ARG(1), true,
                              "ext_oneapi_graph", DEREF(0)),
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
    MEMBER_CALL_FACTORY_ENTRY("cudaGraphExecUpdate", ARG(0), true, "update",
                              DEREF(1)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphExecUpdate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphExecUpdate"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY(
        "cudaGraphCreate",
        CALL(MapNames::getDpctNamespace() + "experimental::create_graph",
             ARG(0), CALL(MapNames::getDpctNamespace() + "get_default_context"),
             CALL(MapNames::getDpctNamespace() + "get_current_device"))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphCreate",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphCreate"),
                            ARG("--use-experimental-features=graph"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph,
    CALL_FACTORY_ENTRY("cudaGraphDestroy",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::destroy_graph",
                            ARG(0))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphDestroy",
                            Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphDestroy"),
                            ARG("--use-experimental-features=graph"))))
