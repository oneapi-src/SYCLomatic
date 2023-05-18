//===--------------- APINamesTemplateType.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

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
    "cudaExtent",
    TYPE_CONDITIONAL_FACTORY(
        CheckTypeNameAndInVarDecl("cudaExtent"),
        CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3")),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3"))))

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
                   HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
                   TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "constant_iterator"), TEMPLATE_ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DplExtrasIterators_arg_index_input_iterator,
    TYPE_REWRITE_ENTRY("cub::ArgIndexInputIterator",
                       HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                           TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "arg_index_input_iterator"),
                                        TEMPLATE_ARG(0)))))

TYPE_REWRITE_ENTRY(
    "cub::DiscardOutputIterator",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
                             TYPE_FACTORY(STR("oneapi::dpl::discard_iterator"))))

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

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DplExtrasIterators_key_value_pair,
    TYPE_REWRITE_ENTRY("cub::KeyValuePair",
                       HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                           TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "key_value_pair"),
                                        TEMPLATE_ARG(0), TEMPLATE_ARG(1)))))

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

// thrust::greater_equal
TYPE_REWRITE_ENTRY(
    "thrust::greater_equal",
    TYPE_FACTORY(STR("std::greater_equal"), TEMPLATE_ARG(0)))

// thrust::less_equal
TYPE_REWRITE_ENTRY(
    "thrust::less_equal",
    TYPE_FACTORY(STR("std::less_equal"), TEMPLATE_ARG(0)))

// logical_and
TYPE_REWRITE_ENTRY(
    "thrust::logical_and",
    TYPE_FACTORY(STR("std::logical_and"), TEMPLATE_ARG(0)))

//  thrust::system_error(
TYPE_REWRITE_ENTRY("thrust::system::system_error",
                   TYPE_FACTORY(STR("std::system_error")))

// thrust::bit_and
TYPE_REWRITE_ENTRY(
    "thrust::bit_and",
    TYPE_FACTORY(STR("std::bit_and"), TEMPLATE_ARG(0)))

// thrust::bit_or
TYPE_REWRITE_ENTRY(
    "thrust::bit_or",
    TYPE_FACTORY(STR("std::bit_or"), TEMPLATE_ARG(0)))

// thrust::minimum
TYPE_REWRITE_ENTRY(
    "thrust::minimum",
    TYPE_FACTORY(STR("oneapi::dpl::minimum"), TEMPLATE_ARG(0)))

//thrust::bit_xor
TYPE_REWRITE_ENTRY(
    "thrust::bit_xor",
    TYPE_FACTORY(STR("std::bit_xor"), TEMPLATE_ARG(0)))

//thrust::modulus
TYPE_REWRITE_ENTRY(
    "thrust::modulus",
    TYPE_FACTORY(STR("std::modulus"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(
    "cub::DoubleBuffer",
    TYPE_CONDITIONAL_FACTORY(
        CheckForPostfixDeclaratorType(0),
        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR),
        HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "io_iterator_pair"), ADD_POINTER(TEMPLATE_ARG(0))))))

TYPE_REWRITE_ENTRY(
    "cudaPos",
    TYPE_CONDITIONAL_FACTORY(
        CheckTypeNameAndInVarDecl("cudaPos"),
        CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3")),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3"))))

TYPE_REWRITE_ENTRY("thrust::identity",
                   TYPE_FACTORY(STR("oneapi::dpl::identity")))

TYPE_REWRITE_ENTRY("thrust::null_type", TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "null_type")))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::plus",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::less",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::greater",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_and",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_and"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_xor",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_xor"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_or",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_or"),
                                TEMPLATE_ARG(0)))

// clang-format on
