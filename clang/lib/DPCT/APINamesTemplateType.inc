//===--------------- APINamesTemplateType.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

TYPE_REWRITE_ENTRY(
    "cuda::atomic",
    TYPE_CONDITIONAL_FACTORY(CheckTemplateArgCount(2),
                             TYPE_FACTORY(STR("atomic_ext"), TEMPLATE_ARG(0),
                                          STR(MapNames::getClNamespace() +
                                              "memory_order::relaxed"),
                                          TEMPLATE_ARG(1)),
                             TYPE_FACTORY(STR("atomic_ext"), TEMPLATE_ARG(0))))