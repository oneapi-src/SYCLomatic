//===--------------- APINamesTemplateType.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

TYPE_REWRITE_ENTRY(
    "cuda::atomic",
        TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cuda::std::atomic",
    TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "thrust::not_equal_to",
    TYPE_FACTORY(STR("std::not_equal_to"), TEMPLATE_ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasIterators_constant_iterator,
TYPE_REWRITE_ENTRY(
    "thrust::constant_iterator",
    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "constant_iterator"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cub::CountingInputIterator",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
                             TYPE_FACTORY(STR("oneapi::dpl::counting_iterator"),
                                          TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY("cub::TransformInputIterator",
                   HEADER_INSERTION_FACTORY(
                       HeaderType::HT_DPL_Iterator,
                       TYPE_FACTORY(STR("oneapi::dpl::transform_iterator"),
                                    TEMPLATE_ARG(2), TEMPLATE_ARG(1))))

TYPE_REWRITE_ENTRY("cub::ConstantInputIterator",
                   HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Utils,
                   TYPE_FACTORY(STR("dpct::constant_iterator"), TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cub::ArgIndexInputIterator",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Utils,
                             TYPE_FACTORY(STR("dpct::arg_index_input_iterator"),
                                          TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY("cub::Sum",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Min",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Max",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Equality", TYPE_FACTORY(STR("std::equal_to"), STR("")))

TYPE_REWRITE_ENTRY(
    "cub::KeyValuePair",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Utils,
                             TYPE_FACTORY(STR("dpct:key_value_pair"),
                                          TEMPLATE_ARG(0), TEMPLATE_ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Memory_usm_host_allocator_alias,
    TYPE_REWRITE_ENTRY("thrust::system::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Memory_usm_host_allocator_alias,
    TYPE_REWRITE_ENTRY("thrust::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))
