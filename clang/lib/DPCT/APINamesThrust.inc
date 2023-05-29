//===--------------- APINamesThrust.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// thrust::replace_copy_if
thrustFactory("thrust::replace_copy_if",
              {{7,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"replace_copy_if", HelperFeatureEnum::DplExtrasAlgorithm_replace_copy_if},
               {6,PolicyState::HasPolicy,3,"oneapi::dpl::replace_copy_if",                 HelperFeatureEnum::no_feature_helper                 },
               {6,PolicyState::NoPolicy ,4,MapNames::getDpctNamespace()+"replace_copy_if", HelperFeatureEnum::DplExtrasAlgorithm_replace_copy_if},
               {5,PolicyState::NoPolicy ,3,"oneapi::dpl::replace_copy_if",                 HelperFeatureEnum::no_feature_helper                 }}),

// thrust::replace_if
thrustFactory("thrust::replace_if",
              {{6,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"replace_if", HelperFeatureEnum::DplExtrasAlgorithm_replace_if},
               {5,PolicyState::HasPolicy,2,"oneapi::dpl::replace_if",                 HelperFeatureEnum::no_feature_helper            },
               {5,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace()+"replace_if", HelperFeatureEnum::DplExtrasAlgorithm_replace_if},
               {4,PolicyState::NoPolicy ,2,"oneapi::dpl::replace_if",                 HelperFeatureEnum::no_feature_helper            }}),

// thrust::remove_copy_if
thrustFactory("thrust::remove_copy_if",
              {{6,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"remove_copy_if", HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if},
               {5,PolicyState::HasPolicy,3,"oneapi::dpl::remove_copy_if",                 HelperFeatureEnum::no_feature_helper                 },
               {5,PolicyState::NoPolicy ,4,MapNames::getDpctNamespace()+"remove_copy_if", HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if},
               {4,PolicyState::NoPolicy ,3,"oneapi::dpl::remove_copy_if",                 HelperFeatureEnum::no_feature_helper                 }}),

// thrust::remove_if
thrustFactory("thrust::remove_if",
              {{5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"remove_if", HelperFeatureEnum::DplExtrasAlgorithm_remove_if},
               {4,PolicyState::HasPolicy,2,"oneapi::dpl::remove_if",                 HelperFeatureEnum::no_feature_helper            },
               {4,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace()+"remove_if", HelperFeatureEnum::DplExtrasAlgorithm_remove_if},
               {3,PolicyState::NoPolicy ,2,"oneapi::dpl::remove_if",                 HelperFeatureEnum::no_feature_helper            }}),

// thrust::not1
CALL_FACTORY_ENTRY("thrust::not1", CALL("oneapi::dpl::not1", ARG(0)))

// thrust::not2
CALL_FACTORY_ENTRY("thrust::not2", CALL("std::not2", ARG(0)))

// thrust::any_of
thrustFactory("thrust::any_of",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::any_of", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,2,"oneapi::dpl::any_of", HelperFeatureEnum::no_feature_helper }}),

// thrust::reverse
thrustFactory("thrust::reverse",
              {{3,PolicyState::HasPolicy,2,"oneapi::dpl::reverse", HelperFeatureEnum::no_feature_helper },
               {2,PolicyState::NoPolicy ,2,"oneapi::dpl::reverse", HelperFeatureEnum::no_feature_helper }}),

// thrust::find_if
thrustFactory("thrust::find_if",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::find_if", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,2,"oneapi::dpl::find_if", HelperFeatureEnum::no_feature_helper }}),

// thrust::find_if_not
thrustFactory("thrust::find_if_not",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::find_if_not", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,2,"oneapi::dpl::find_if_not", HelperFeatureEnum::no_feature_helper }}),

// thrust::replace
thrustFactory("thrust::replace",
              {{5,PolicyState::HasPolicy,2,"oneapi::dpl::replace", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::NoPolicy ,2,"oneapi::dpl::replace", HelperFeatureEnum::no_feature_helper }}),

// thrust::replace_copy
thrustFactory("thrust::replace_copy",
              {{6,PolicyState::HasPolicy,3,"oneapi::dpl::replace_copy", HelperFeatureEnum::no_feature_helper },
               {5,PolicyState::NoPolicy ,3,"oneapi::dpl::replace_copy", HelperFeatureEnum::no_feature_helper }}),

// thrust::adjacent_difference
thrustFactory("thrust::adjacent_difference",
              {{5,PolicyState::HasPolicy,3,"oneapi::dpl::adjacent_difference", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::HasPolicy,3,"oneapi::dpl::adjacent_difference", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::NoPolicy ,3,"oneapi::dpl::adjacent_difference", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,3,"oneapi::dpl::adjacent_difference", HelperFeatureEnum::no_feature_helper }}),

// thrust::mismatch
thrustFactory("thrust::mismatch",
              {{5,PolicyState::HasPolicy,3,"oneapi::dpl::mismatch", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::HasPolicy,3,"oneapi::dpl::mismatch", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::NoPolicy ,3,"oneapi::dpl::mismatch", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,3,"oneapi::dpl::mismatch", HelperFeatureEnum::no_feature_helper }}),

// thrust::gather_if
thrustFactory("thrust::gather_if",
              {{7,PolicyState::HasPolicy,5,MapNames::getDpctNamespace()+"gather_if", HelperFeatureEnum::DplExtrasAlgorithm_gather_if},
               {6,PolicyState::NoPolicy ,5,MapNames::getDpctNamespace()+"gather_if", HelperFeatureEnum::DplExtrasAlgorithm_gather_if}}),

// thrust::merge_by_key
thrustFactory("thrust::merge_by_key",
              {{10,PolicyState::HasPolicy,8,MapNames::getDpctNamespace()+"merge", HelperFeatureEnum::DplExtrasAlgorithm_merge},
               { 9,PolicyState::HasPolicy,8,MapNames::getDpctNamespace()+"merge", HelperFeatureEnum::DplExtrasAlgorithm_merge},
               { 9,PolicyState::NoPolicy ,8,MapNames::getDpctNamespace()+"merge", HelperFeatureEnum::DplExtrasAlgorithm_merge},
               { 8,PolicyState::NoPolicy ,8,MapNames::getDpctNamespace()+"merge", HelperFeatureEnum::DplExtrasAlgorithm_merge}}),

// thrust::inclusive_scan
thrustFactory("thrust::inclusive_scan",
              {{5,PolicyState::HasPolicy,3,"oneapi::dpl::inclusive_scan", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::HasPolicy,3,"oneapi::dpl::inclusive_scan", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::NoPolicy ,3,"oneapi::dpl::inclusive_scan", HelperFeatureEnum::no_feature_helper },
               {3,PolicyState::NoPolicy ,3,"oneapi::dpl::inclusive_scan", HelperFeatureEnum::no_feature_helper }}),

// thrust::gather
thrustFactory("thrust::gather",
              {{5,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"gather", HelperFeatureEnum::DplExtrasAlgorithm_gather },
               {4,PolicyState::NoPolicy ,4,MapNames::getDpctNamespace()+"gather", HelperFeatureEnum::DplExtrasAlgorithm_gather }}),

// thrust::scatter
thrustFactory("thrust::scatter",
              {{5,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"scatter", HelperFeatureEnum::DplExtrasAlgorithm_scatter },
               {4,PolicyState::NoPolicy ,4,MapNames::getDpctNamespace()+"scatter", HelperFeatureEnum::DplExtrasAlgorithm_scatter }}),

// thrust::transform_if
thrustFactory("thrust::transform_if",
              {{6,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if},
               {5,PolicyState::NoPolicy,3,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if},
               {7,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if},
               {6,PolicyState::NoPolicy,4,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if},
               {8,PolicyState::HasPolicy,4,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if},
               {7,PolicyState::NoPolicy,4,MapNames::getDpctNamespace()+"transform_if", HelperFeatureEnum::DplExtrasAlgorithm_transform_if}}),


// thrust::// thrust::unique_by_key_copy
thrustFactory("thrust::unique_by_key_copy",
              {{6,PolicyState::HasPolicy,5,MapNames::getDpctNamespace()+"unique_copy", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {5,PolicyState::NoPolicy, 5,MapNames::getDpctNamespace()+"unique_copy", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {7,PolicyState::HasPolicy,5,MapNames::getDpctNamespace()+"unique_copy", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {6,PolicyState::NoPolicy, 5,MapNames::getDpctNamespace()+"unique_copy", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy}}),


// As the order of the last three parameters are changed in the mapping api, so just keep it.
// thrust::transform_reduce
CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(0, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY(
            "thrust::transform_reduce",
            CALL("std::transform_reduce",
                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                 ARG(0), ARG(1), ARG(3), ARG(4), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::transform_reduce",
                           CALL("std::transform_reduce",
                                ARG("oneapi::dpl::execution::seq"), ARG(0),
                                ARG(1), ARG(3), ARG(4), ARG(2)))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(0, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY(
            "thrust::transform_reduce",
            CALL("std::transform_reduce",
                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                 ARG(1), ARG(2), ARG(4), ARG(5), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::transform_reduce",
                           CALL("std::transform_reduce",
                                ARG("oneapi::dpl::execution::seq"), ARG(1),
                                ARG(2), ARG(4), ARG(5), ARG(3)))))


// thrust::swap
CALL_FACTORY_ENTRY("thrust::swap",  CALL("std::swap",  ARG(0), ARG(1)))

// thrust::make_pair
CALL_FACTORY_ENTRY("thrust::make_pair",  CALL("std::make_pair",  ARG(0), ARG(1)))

// thrust::stable_sort_by_key
thrustFactory("thrust::stable_sort_by_key",
              {{5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "stable_sort", HelperFeatureEnum::DplExtrasAlgorithm_stable_sort},
               {4,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "stable_sort", HelperFeatureEnum::DplExtrasAlgorithm_stable_sort},
               {4,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace() + "stable_sort", HelperFeatureEnum::DplExtrasAlgorithm_stable_sort},
               {3,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace() + "stable_sort", HelperFeatureEnum::DplExtrasAlgorithm_stable_sort}}),


// thrust::find
thrustFactory("thrust::find",
              {{4,PolicyState::HasPolicy,4,"oneapi::dpl::find", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy ,4,"oneapi::dpl::find", HelperFeatureEnum::no_feature_helper}}),

// thrust::sort_by_key
thrustFactory("thrust::sort_by_key",
              {{5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "sort", HelperFeatureEnum::DplExtrasAlgorithm_sort},
               {4,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "sort", HelperFeatureEnum::DplExtrasAlgorithm_sort},
               {4,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace() + "sort", HelperFeatureEnum::DplExtrasAlgorithm_sort},
               {3,PolicyState::NoPolicy ,3,MapNames::getDpctNamespace() + "sort", HelperFeatureEnum::DplExtrasAlgorithm_sort}}),

// thrust::inner_product
thrustFactory("thrust::inner_product",
              {{5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"inner_product", HelperFeatureEnum::DplExtrasNumeric_inner_product},
               {4,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace()+"inner_product", HelperFeatureEnum::DplExtrasNumeric_inner_product},
               {7,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"inner_product", HelperFeatureEnum::DplExtrasNumeric_inner_product},
               {6,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace()+"inner_product", HelperFeatureEnum::DplExtrasNumeric_inner_product}}),


// thrust::reduce_by_key
thrustFactory("thrust::reduce_by_key",
              {{6,PolicyState::HasPolicy,5,"oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 5, "oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,5,"oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 5, "oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper},
               {8,PolicyState::HasPolicy,5,"oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::NoPolicy, 5, "oneapi::dpl::reduce_by_segment", HelperFeatureEnum::no_feature_helper}}),

// thrust::for_each
CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    // Handling case: thrust::for_each(policy, device.begin(), device.end(), f);
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(
            makeCheckAnd(CheckThrustArgType(0, "thrust::cuda_cub::par_nosync"),
                         checkEnableExtDPLAPI()),
            CheckThrustArgType(1, "thrust::device_ptr")),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Async,
            CALL_FACTORY_ENTRY(
                "thrust::for_each",
                CALL("oneapi::dpl::experimental::for_each_async",
                     CALL("oneapi::dpl::execution::make_device_policy",
                          QUEUESTR),
                     ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
        CALL_FACTORY_ENTRY("thrust::for_each",
                           CALL("std::for_each", makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
    // Handling case: thrust::for_each(host.begin(), host.end(), f);
    CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(0, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY(
            "thrust::for_each",
            CALL("std::for_each",
                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                 ARG(0), ARG(1), THRUST_FUNCTOR(2))),
        CALL_FACTORY_ENTRY("thrust::for_each",
                           CALL("std::for_each",
                                ARG("oneapi::dpl::execution::seq"), ARG(0),
                                ARG(1), THRUST_FUNCTOR(2)))))

// thrust::transform
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  //Handling case: thrust::transform(policy, device.begin(), device.end(), device.end(), device.end(), f);
  CALL_FACTORY_ENTRY("thrust::transform",
                      CALL("std::transform",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), THRUST_FUNCTOR(5))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    //Handling case: thrust::transform(host.begin(), host.end(), host.end(), f);
    CONDITIONAL_FACTORY_ENTRY(
      CheckThrustArgType(0, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3))),
      CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
      CONDITIONAL_FACTORY_ENTRY(
        IsPolicyArgType(0),
        //Handling case: thrust::transform(policy, device.begin(), device.end(), device.begin(), f);
        CALL_FACTORY_ENTRY("thrust::transform",
                        CALL("std::transform",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
        //Handling case: thrust::transform(device.begin(), device.end(), device.begin(), device.end(), f);
        CONDITIONAL_FACTORY_ENTRY(
          CheckThrustArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
          CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))))
      ))
)

// thrust::copy_if
thrustFactory("thrust::copy_if",
              {{5,PolicyState::HasPolicy,3,"std::copy_if", HelperFeatureEnum::no_feature_helper },
               {4,PolicyState::NoPolicy, 3,"std::copy_if", HelperFeatureEnum::no_feature_helper },
               {6,PolicyState::HasPolicy,4, MapNames::getDpctNamespace() + "copy_if", HelperFeatureEnum::DplExtrasAlgorithm_copy_if },
               {5,PolicyState::NoPolicy ,4, MapNames::getDpctNamespace() + "copy_if", HelperFeatureEnum::DplExtrasAlgorithm_copy_if }}),

// thrust::make_zip_iterator
CALL_FACTORY_ENTRY("thrust::make_zip_iterator",  CALL("oneapi::dpl::make_zip_iterator",  ARG(0)))


// thrust::inclusive_scan_by_key
thrustFactory("thrust::inclusive_scan_by_key",
              {{5,PolicyState::HasPolicy,4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::HasPolicy,4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 4,"oneapi::dpl::inclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper}}),

// thrust::exclusive_scan_by_key
thrustFactory("thrust::exclusive_scan_by_key",
              {{5,PolicyState::HasPolicy,4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::HasPolicy,4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {8,PolicyState::HasPolicy,4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::NoPolicy, 4,"oneapi::dpl::exclusive_scan_by_segment", HelperFeatureEnum::no_feature_helper}}),


// thrust::make_reverse_iterator
CALL_FACTORY_ENTRY("thrust::make_reverse_iterator",  CALL("oneapi::dpl::make_reverse_iterator",  ARG(0)))

// thrust::exclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(3),
  // thrust::exclusive_scan(data, data + 6, data)
  CONDITIONAL_FACTORY_ENTRY(
    CheckThrustArgType(1, "thrust::device_ptr"),
    CALL_FACTORY_ENTRY("thrust::exclusive_scan",
      CALL("std::exclusive_scan",
        CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
        ARG(0), ARG(1), ARG(2), makeAddTypeCastThrustToZero(2))),
    CALL_FACTORY_ENTRY("thrust::exclusive_scan",
      CALL("std::exclusive_scan",
        ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), makeAddTypeCastThrustToZero(2)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      IsPolicyArgType(0),
      // thrust::exclusive_scan(thrust::host, data, data + 6, data);
      CALL_FACTORY_ENTRY("thrust::exclusive_scan",
        CALL("std::exclusive_scan",
          makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), makeAddTypeCastThrustToZero(3))),
      // thrust::exclusive_scan(data, data + 6, data, 4);
      CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
            ARG(0), ARG(1), ARG(2), makeAddTypeCastToThrustArg(3))),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), makeAddTypeCastToThrustArg(3)))
      )
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(5),
      CONDITIONAL_FACTORY_ENTRY(
        IsPolicyArgType(0),
        // thrust::exclusive_scan(thrust::host, data, data + 6, data, 4);
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), makeAddTypeCastToThrustArg(4))),
        // thrust::exclusive_scan(data, data + 10, data, 1, binary_op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckThrustArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::exclusive_scan",
            CALL("std::exclusive_scan",
              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
              ARG(0), ARG(1), ARG(2), makeAddTypeCastToThrustArg(3), ARG(4))),
          CALL_FACTORY_ENTRY("thrust::exclusive_scan",
            CALL("std::exclusive_scan",
              ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), makeAddTypeCastToThrustArg(3), ARG(4)))
        )
      ),
      // thrust::exclusive_scan(thrust::host, data, data + 10, data, 1, binary_op);
      CALL_FACTORY_ENTRY("thrust::exclusive_scan",
        CALL("std::exclusive_scan",
          makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), makeAddTypeCastToThrustArg(4), ARG(5)))
    )
  )
)

// thrust::get
CALL_FACTORY_ENTRY(
    "thrust::get",
    CALL(TEMPLATED_CALLEE("std::get", std::vector<size_t>(1, 0)), ARG(0)))

// thrust::unique_by_key
thrustFactory("thrust::unique_by_key",
              {{4,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"unique", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {3,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace()+"unique", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace()+"unique", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {4,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace()+"unique", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy}}),

// thrust::log10
CALL_FACTORY_ENTRY("thrust::log10", CALL("std::log10", ARG(0)))

// thrust::sqrt
CALL_FACTORY_ENTRY("thrust::sqrt", CALL("std::sqrt", ARG(0)))

// thrust::pow
CALL_FACTORY_ENTRY("thrust::pow", CALL("std::pow", ARG(0), ARG(1)))

// thrust::sin
CALL_FACTORY_ENTRY("thrust::sin", CALL("std::sin", ARG(0)))

// thrust::cos
CALL_FACTORY_ENTRY("thrust::cos", CALL("std::cos", ARG(0)))

// thrust::tan
CALL_FACTORY_ENTRY("thrust::tan", CALL("std::tan", ARG(0)))

// thrust::asin
CALL_FACTORY_ENTRY("thrust::asin", CALL("std::asin", ARG(0)))

// thrust::acos
CALL_FACTORY_ENTRY("thrust::acos", CALL("std::acos", ARG(0)))

// thrust::atan
CALL_FACTORY_ENTRY("thrust::atan", CALL("std::atan", ARG(0)))

// thrust::sinh
CALL_FACTORY_ENTRY("thrust::sinh", CALL("std::sinh", ARG(0)))

// thrust::cosh
CALL_FACTORY_ENTRY("thrust::cosh", CALL("std::cosh", ARG(0)))

// thrust::tanh
CALL_FACTORY_ENTRY("thrust::tanh", CALL("std::tanh", ARG(0)))

// thrust::asinh
CALL_FACTORY_ENTRY("thrust::asinh", CALL("std::asinh", ARG(0)))

// thrust::acosh
CALL_FACTORY_ENTRY("thrust::acosh", CALL("std::acosh", ARG(0)))

// thrust::atanh
CALL_FACTORY_ENTRY("thrust::atanh", CALL("std::atanh", ARG(0)))

// thrust::abs
CALL_FACTORY_ENTRY("thrust::abs", CALL("std::abs", ARG(0)))

// thrust::polar
CALL_FACTORY_ENTRY("thrust::polar", CALL("std::polar", ARG(0), ARG(1)))

// thrust::exp
CALL_FACTORY_ENTRY("thrust::exp", CALL("std::exp", ARG(0)))

// thrust::log
CALL_FACTORY_ENTRY("thrust::log", CALL("std::log", ARG(0)))

// thrust::distance
CALL_FACTORY_ENTRY("thrust::distance",
                   CALL("oneapi::dpl::distance", ARG(0), ARG(1)))

// thrust::make_permutation_iterator
CALL_FACTORY_ENTRY("thrust::make_permutation_iterator",
                   CALL("oneapi::dpl::make_permutation_iterator", ARG(0),
                        ARG(1)))

// thrust::make_transform_iterator
CALL_FACTORY_ENTRY("thrust::make_transform_iterator",
                   CALL("oneapi::dpl::make_transform_iterator", ARG(0),
                        ARG(1)))

// thrust::norm
CALL_FACTORY_ENTRY("thrust::norm", CALL("std::norm", ARG(0)))

// thrust::conj
CALL_FACTORY_ENTRY("thrust::conj", CALL("std::conj", ARG(0)))

// thrust::proj
CALL_FACTORY_ENTRY("thrust::proj", CALL("std::proj", ARG(0)))

// thrust::raw_reference_cast
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_get_raw_reference,
CALL_FACTORY_ENTRY("thrust::raw_reference_cast", CALL(MapNames::getDpctNamespace() + "get_raw_reference", ARG(0))))

// thrust::minmax_element
thrustFactory("thrust::minmax_element",
              {{3,PolicyState::HasPolicy,2,"oneapi::dpl::minmax_element", HelperFeatureEnum::no_feature_helper},
               {2,PolicyState::NoPolicy, 2,"oneapi::dpl::minmax_element", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::HasPolicy,2,"oneapi::dpl::minmax_element", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::minmax_element", HelperFeatureEnum::no_feature_helper}}),

// thrust::is_sorted
thrustFactory("thrust::is_sorted",
              {{3,PolicyState::HasPolicy,2,"oneapi::dpl::is_sorted", HelperFeatureEnum::no_feature_helper},
               {2,PolicyState::NoPolicy, 2,"oneapi::dpl::is_sorted", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::HasPolicy,2,"oneapi::dpl::is_sorted", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::is_sorted", HelperFeatureEnum::no_feature_helper}}),

// thrust::partition
thrustFactory("thrust::partition",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::partition", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::partition", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "partition", HelperFeatureEnum::DplExtrasAlgorithm_partition},
               {4,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace() + "partition", HelperFeatureEnum::DplExtrasAlgorithm_partition}}),


// thrust::unique_copy
thrustFactory("thrust::unique_copy",
              {{4,PolicyState::HasPolicy,3,"oneapi::dpl::unique_copy", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 3,"oneapi::dpl::unique_copy", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,3,"oneapi::dpl::unique_copy", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 3,"oneapi::dpl::unique_copy", HelperFeatureEnum::no_feature_helper}}),

// thrust::stable_sort
thrustFactory("thrust::stable_sort",
              {{3,PolicyState::HasPolicy,2,"oneapi::dpl::stable_sort", HelperFeatureEnum::no_feature_helper},
               {2,PolicyState::NoPolicy, 2,"oneapi::dpl::stable_sort", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::HasPolicy,2,"oneapi::dpl::stable_sort", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::stable_sort", HelperFeatureEnum::no_feature_helper}}),

// thrust::set_difference_by_key
thrustFactory("thrust::set_difference_by_key",
              {{ 9,PolicyState::HasPolicy,8,MapNames::getDpctNamespace()+"set_difference", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               { 8,PolicyState::NoPolicy, 8,MapNames::getDpctNamespace()+"set_difference", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               {10,PolicyState::HasPolicy,8,MapNames::getDpctNamespace()+"set_difference", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy},
               { 9,PolicyState::NoPolicy, 8,MapNames::getDpctNamespace()+"set_difference", HelperFeatureEnum::DplExtrasAlgorithm_unique_copy}}),

// thrust::set_difference
thrustFactory("thrust::set_difference",
              {{ 6,PolicyState::HasPolicy,5,"oneapi::dpl::set_difference", HelperFeatureEnum::no_feature_helper},
               { 5,PolicyState::NoPolicy, 5,"oneapi::dpl::set_difference", HelperFeatureEnum::no_feature_helper},
               { 7,PolicyState::HasPolicy,5,"oneapi::dpl::set_difference", HelperFeatureEnum::no_feature_helper},
               { 6,PolicyState::NoPolicy, 5,"oneapi::dpl::set_difference", HelperFeatureEnum::no_feature_helper}}),


// thrust::remove_copy
thrustFactory("thrust::remove_copy",
              {{5,PolicyState::HasPolicy,3,"oneapi::dpl::remove_copy", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy ,3,"oneapi::dpl::remove_copy", HelperFeatureEnum::no_feature_helper}}),


// thrust::transform_exclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(7),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::transform_exclusive_scan(policy, ptr, ptr, ptr, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::transform_exclusive_scan",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                          CALL("oneapi::dpl::transform_exclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(5), ARG(6), ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                        CALL("oneapi::dpl::transform_exclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(5), ARG(6), ARG(4)))),
    // Handling case: thrust::unique_by_key(thrust::device, iterator, iterator, iterator, val);
    CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                      CALL("oneapi::dpl::transform_exclusive_scan",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(5), ARG(6), ARG(4)))
  ),

    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::transform_exclusive_scan(ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::transform_exclusive_scan",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(0)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                            CALL("oneapi::dpl::transform_exclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(4), ARG(5), ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                          CALL("oneapi::dpl::transform_exclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(4), ARG(5), ARG(3)))),
      CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::transform_exclusive_scan(iterator, iterator, iterator, iterator);
        CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                            CALL("oneapi::dpl::transform_exclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(4), ARG(5), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::transform_exclusive_scan",
                            CALL("oneapi::dpl::transform_exclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(4), ARG(5), ARG(3))))
    )
)


// thrust::transform_inclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::transform_inclusive_scan(policy, ptr, ptr, ptr, unary_op, binary_op);
    IFELSE_FACTORY_ENTRY(
      "thrust::transform_inclusive_scan",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                          CALL("oneapi::dpl::transform_inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(5), ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                        CALL("oneapi::dpl::transform_inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(5), ARG(4)))),
    // Handling case: thrust::unique_by_key(thrust::device, iterator, iterator, iterator, unary_op, binary_op);
    CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                      CALL("oneapi::dpl::transform_inclusive_scan",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(5), ARG(4)))
  ),

    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::transform_inclusive_scan(ptr, ptr, ptr, unary_op, binary_op);
      IFELSE_FACTORY_ENTRY(
        "thrust::transform_inclusive_scan",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(0)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                            CALL("oneapi::dpl::transform_inclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(4), ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                          CALL("oneapi::dpl::transform_inclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(4), ARG(3)))),
      CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::transform_inclusive_scan(iterator, iterator, iterator, unary_op, binary_op);
        CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                            CALL("oneapi::dpl::transform_inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(4), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::transform_inclusive_scan",
                            CALL("oneapi::dpl::transform_inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(4), ARG(3))))
    )
)


// thrust::tabulate
thrustFactory("thrust::tabulate",
              {{4,PolicyState::HasPolicy,2,MapNames::getDpctNamespace() + "for_each_index", HelperFeatureEnum::DplExtrasAlgorithm_partition},
               {3,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace() + "for_each_index", HelperFeatureEnum::DplExtrasAlgorithm_partition}}),



// thrust::for_each_n
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::for_each_n(policy, ptr, ptr, ptr, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::for_each_n",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::for_each_n", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::for_each_n",
                          CALL("oneapi::dpl::for_each_n",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                ARG(2), THRUST_FUNCTOR(3)))),
      CALL_FACTORY_ENTRY("thrust::for_each_n",
                        CALL("oneapi::dpl::for_each_n",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
    // Handling case: thrust::unique_by_key(thrust::device, iterator, iterator, iterator, val);
    CALL_FACTORY_ENTRY("thrust::for_each_n",
                      CALL("oneapi::dpl::for_each_n",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), THRUST_FUNCTOR(3)))
  ),

    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(0), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::for_each_n(ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::for_each_n",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::for_each_n", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::for_each_n",
                            CALL("oneapi::dpl::for_each_n",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  ARG(1), THRUST_FUNCTOR(2)))),
        CALL_FACTORY_ENTRY("thrust::for_each_n",
                          CALL("oneapi::dpl::for_each_n",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), THRUST_FUNCTOR(2)))),
      CONDITIONAL_FACTORY_ENTRY(
        CheckThrustArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::for_each_n(iterator, iterator, iterator);
        CALL_FACTORY_ENTRY("thrust::for_each_n",
                            CALL("oneapi::dpl::for_each_n",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), THRUST_FUNCTOR(2))),
        CALL_FACTORY_ENTRY("thrust::for_each_n",
                            CALL("oneapi::dpl::for_each_n",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), THRUST_FUNCTOR(2))))
    )
)


// thrust::set_intersection_by_key
thrustFactory("thrust::set_intersection_by_key",
              {{8,PolicyState::HasPolicy,7,MapNames::getDpctNamespace()+"set_intersection", HelperFeatureEnum::DplExtrasAlgorithm_set_intersection},
               {7,PolicyState::NoPolicy, 7,MapNames::getDpctNamespace()+"set_intersection", HelperFeatureEnum::DplExtrasAlgorithm_set_intersection},
               {9,PolicyState::HasPolicy,7,MapNames::getDpctNamespace()+"set_intersection", HelperFeatureEnum::DplExtrasAlgorithm_set_intersection},
               {8,PolicyState::NoPolicy, 7,MapNames::getDpctNamespace()+"set_intersection", HelperFeatureEnum::DplExtrasAlgorithm_set_intersection}}),



// thrust::partition_copy
thrustFactory("thrust::partition_copy",
              {{6,PolicyState::HasPolicy,4,"oneapi::dpl::partition_copy", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 4,"oneapi::dpl::partition_copy", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,5,MapNames::getDpctNamespace() + "partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_partition},
               {6,PolicyState::NoPolicy, 5,MapNames::getDpctNamespace() + "partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_partition}}),


// thrust::stable_partition_copy
thrustFactory("thrust::stable_partition_copy",
              {{6,PolicyState::HasPolicy,4,MapNames::getDpctNamespace() + "stable_partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_stable_partition_copy},
               {5,PolicyState::NoPolicy, 4,MapNames::getDpctNamespace() + "stable_partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_stable_partition_copy},
               {7,PolicyState::HasPolicy,5,MapNames::getDpctNamespace() + "stable_partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_stable_partition_copy},
               {6,PolicyState::NoPolicy, 5,MapNames::getDpctNamespace() + "stable_partition_copy", HelperFeatureEnum::DplExtrasAlgorithm_stable_partition_copy}}),



// thrust::stable_partition
thrustFactory("thrust::stable_partition",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::stable_partition", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::stable_partition", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,3,MapNames::getDpctNamespace() + "stable_partition", HelperFeatureEnum::DplExtrasAlgorithm_partition},
               {4,PolicyState::NoPolicy, 3,MapNames::getDpctNamespace() + "stable_partition", HelperFeatureEnum::DplExtrasAlgorithm_partition}}),

// thrust::remove
thrustFactory("thrust::remove",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::remove", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::remove", HelperFeatureEnum::no_feature_helper}}),

// thrust::copy
thrustFactory("thrust::copy",
              {{4,PolicyState::HasPolicy,3,"std::copy", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 3,"std::copy", HelperFeatureEnum::no_feature_helper}}),



// thrust::copy_n
CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      // Handle case thrust::copy(ptr, N, des_ptr)
        makeCheckAnd(CheckIsPtr(0), makeCheckNot(checkIsUSM())),
            IFELSE_FACTORY_ENTRY("thrust::copy_n",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
                CALL_FACTORY_ENTRY("thrust::copy_n", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(0)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
                CALL_FACTORY_ENTRY("thrust::copy_n",
                            CALL("std::copy_n",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  ARG(1),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
            CALL_FACTORY_ENTRY("thrust::copy_n",
                          CALL("std::copy_n",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2)))),
        CONDITIONAL_FACTORY_ENTRY(
            CheckThrustArgType(0, "thrust::device_ptr"),
            // Handle case thrust::copy(dev_itr, N, dev_itr)
            CALL_FACTORY_ENTRY("thrust::copy_n",
                CALL("std::copy_n",
                     CALL("oneapi::dpl::execution::make_device_policy",
                          QUEUESTR),
                     ARG(0), ARG(1), ARG(2))),
            // Handle case thrust::copy(host_itr, N, host_itr)
            CALL_FACTORY_ENTRY("thrust::copy_n",
                               CALL("std::copy_n",
                                    ARG("oneapi::dpl::execution::seq"), ARG(0),
                                    ARG(1), ARG(2))))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
            IFELSE_FACTORY_ENTRY("thrust::copy_n",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
                CALL_FACTORY_ENTRY("thrust::copy_n", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
                CALL_FACTORY_ENTRY("thrust::copy_n",
                            CALL("std::copy_n",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  ARG(2),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
            CALL_FACTORY_ENTRY("thrust::copy_n",
                          CALL("std::copy_n",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3)))),
        // Handle case thrust::copy(policy,itr, N, itr)
        CALL_FACTORY_ENTRY("thrust::copy_n",
                           CALL("std::copy_n", makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3)))))

// thrust::cuda_category
CALL_FACTORY_ENTRY("thrust::cuda_category", CALL("std::generic_category"))

// thrust::equal_range
thrustFactory("thrust::equal_range",
              {{4,PolicyState::HasPolicy,2,MapNames::getDpctNamespace()+"equal_range", HelperFeatureEnum::DplExtrasAlgorithm_equal_range},
               {3,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace()+"equal_range", HelperFeatureEnum::DplExtrasAlgorithm_equal_range},
               {5,PolicyState::HasPolicy,2,MapNames::getDpctNamespace()+"equal_range", HelperFeatureEnum::DplExtrasAlgorithm_equal_range},
               {4,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace()+"equal_range", HelperFeatureEnum::DplExtrasAlgorithm_equal_range}}),
// thrust::sequence
thrustFactory("thrust::sequence",
              {{3,PolicyState::HasPolicy,2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota},
               {2,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota},
               {4,PolicyState::HasPolicy,2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota},
               {3,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota},
               {5,PolicyState::HasPolicy,2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota},
               {6,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace()+"iota", HelperFeatureEnum::DplExtrasAlgorithm_iota}}),

// thrust::sort
thrustFactory("thrust::sort",
              {{ 3,PolicyState::HasPolicy,2,"oneapi::dpl::sort", HelperFeatureEnum::no_feature_helper},
               { 2,PolicyState::NoPolicy, 2,"oneapi::dpl::sort", HelperFeatureEnum::no_feature_helper},
               { 4,PolicyState::HasPolicy,2,"oneapi::dpl::sort", HelperFeatureEnum::no_feature_helper},
               { 3,PolicyState::NoPolicy, 2,"oneapi::dpl::sort", HelperFeatureEnum::no_feature_helper}}),

// thrust::reduce
thrustFactory("thrust::reduce",
              {{3,PolicyState::HasPolicy,2,"std::reduce", HelperFeatureEnum::no_feature_helper},
               {2,PolicyState::NoPolicy, 2,"std::reduce", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::HasPolicy,2,"std::reducet", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::reduce", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,2,"std::reduce", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 2,"std::reduce", HelperFeatureEnum::no_feature_helper}}),

// thrust::device_free
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_free_device,
CALL_FACTORY_ENTRY("thrust::device_free", CALL(MapNames::getDpctNamespace() + "free_device", ARG(0))))


// thrust::device_malloc
// CALL_FACTORY_ENTRY("thrust::device_malloc", CALL(MapNames::getDpctNamespace() + "device_malloc", ARG(0)))

// thrust::raw_pointer_cast
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_get_raw_pointer,
CALL_FACTORY_ENTRY("thrust::raw_pointer_cast", CALL(MapNames::getDpctNamespace() + "get_raw_pointer", ARG(0))))

// thrust::count_if
thrustFactory("thrust::count_if",
              {{4,PolicyState::HasPolicy,2,"std::count_if", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::count_if", HelperFeatureEnum::no_feature_helper}}),

// thrust::fill
thrustFactory("thrust::fill",
              {{4,PolicyState::HasPolicy,2,"std::fill", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::fill", HelperFeatureEnum::no_feature_helper}}),

// thrust::fill_n
thrustFactory("thrust::fill_n",
              {{4,PolicyState::HasPolicy,1,"std::fill_n", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 1,"std::fill_n", HelperFeatureEnum::no_feature_helper}}),

// thrust::partition_point
thrustFactory("thrust::partition_point",
              {{4,PolicyState::HasPolicy,2,MapNames::getDpctNamespace() + "partition_point", HelperFeatureEnum::DplExtrasAlgorithm_partition_point},
               {3,PolicyState::NoPolicy, 2,MapNames::getDpctNamespace() + "partition_point", HelperFeatureEnum::DplExtrasAlgorithm_partition_point}}),

// thrust::binary_search
thrustFactory("thrust::binary_search",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,2,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 2,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::HasPolicy,5,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 5,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,5,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 5,"oneapi::dpl::binary_search", HelperFeatureEnum::no_feature_helper}}),

// thrust::lower_bound
thrustFactory("thrust::lower_bound",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,2,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 2,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::HasPolicy,5,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 5,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,5,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 5,"oneapi::dpl::lower_bound", HelperFeatureEnum::no_feature_helper}}),

// thrust::upper_bound
thrustFactory("thrust::upper_bound",
              {{4,PolicyState::HasPolicy,2,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::HasPolicy,2,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {4,PolicyState::NoPolicy, 2,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::HasPolicy,5,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {5,PolicyState::NoPolicy, 5,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {7,PolicyState::HasPolicy,5,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper},
               {6,PolicyState::NoPolicy, 5,"oneapi::dpl::upper_bound", HelperFeatureEnum::no_feature_helper}}),

// thrust::count
thrustFactory("thrust::count",
              {{4,PolicyState::HasPolicy,2,"std::count", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::count", HelperFeatureEnum::no_feature_helper}}),

// thrust::generate
thrustFactory("thrust::generate",
              {{4,PolicyState::HasPolicy,2,"std::generate", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::generate", HelperFeatureEnum::no_feature_helper}}),

// thrust::generate_n
thrustFactory("thrust::generate_n",
              {{4,PolicyState::HasPolicy,2,"std::generate_n", HelperFeatureEnum::no_feature_helper},
               {3,PolicyState::NoPolicy, 2,"std::generate_n", HelperFeatureEnum::no_feature_helper}}),

// thrust::merge
thrustFactory("thrust::merge",
              {{ 6,PolicyState::HasPolicy,5,"std::merge", HelperFeatureEnum::no_feature_helper},
               { 5,PolicyState::NoPolicy, 5,"std::merge", HelperFeatureEnum::no_feature_helper},
               { 7,PolicyState::HasPolicy,5,"std::merge", HelperFeatureEnum::no_feature_helper},
               { 6,PolicyState::NoPolicy, 5,"std::merge", HelperFeatureEnum::no_feature_helper}}),

// thrust::uninitialized_fill
// thrustFactory("thrust::uninitialized_fill",
//               {{4,PolicyState::HasPolicy,2,"std::uninitialized_fill", HelperFeatureEnum::no_feature_helper},
//                {3,PolicyState::NoPolicy, 2,"std::uninitialized_fill", HelperFeatureEnum::no_feature_helper}}),

// thrust::unique
thrustFactory("thrust::unique",
              {{ 3,PolicyState::HasPolicy,2,"std::unique", HelperFeatureEnum::no_feature_helper},
               { 2,PolicyState::NoPolicy, 2,"std::unique", HelperFeatureEnum::no_feature_helper},
               { 4,PolicyState::HasPolicy,2,"std::unique", HelperFeatureEnum::no_feature_helper},
               { 3,PolicyState::NoPolicy, 2,"std::unique", HelperFeatureEnum::no_feature_helper}}),

// thrust::max_element
thrustFactory("thrust::max_element",
              {{ 3,PolicyState::HasPolicy,2,"std::max_element", HelperFeatureEnum::no_feature_helper},
               { 2,PolicyState::NoPolicy, 2,"std::max_element", HelperFeatureEnum::no_feature_helper},
               { 4,PolicyState::HasPolicy,2,"std::max_element", HelperFeatureEnum::no_feature_helper},
               { 3,PolicyState::NoPolicy, 2,"std::max_element", HelperFeatureEnum::no_feature_helper}}),

// thrust::min_element
thrustFactory("thrust::min_element",
              {{ 3,PolicyState::HasPolicy,2,"std::min_element", HelperFeatureEnum::no_feature_helper},
               { 2,PolicyState::NoPolicy, 2,"std::min_element", HelperFeatureEnum::no_feature_helper},
               { 4,PolicyState::HasPolicy,2,"std::min_element", HelperFeatureEnum::no_feature_helper},
               { 3,PolicyState::NoPolicy, 2,"std::min_element", HelperFeatureEnum::no_feature_helper}}),

// thrust::make_discard_iterator
CALL_FACTORY_ENTRY("thrust::make_discard_iterator", CALL("oneapi::dpl::discard_iterator"))

// thrust::make_constant_iterator
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasIterators_make_constant_iterator,
CALL_FACTORY_ENTRY("thrust::make_constant_iterator", CALL(MapNames::getDpctNamespace() + "make_constant_iterator", ARG(0))))

// thrust::make_counting_iterator
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasIterators_make_counting_iterator,
CALL_FACTORY_ENTRY("thrust::make_counting_iterator", CALL(MapNames::getDpctNamespace() + "make_counting_iterator", ARG(0))))
