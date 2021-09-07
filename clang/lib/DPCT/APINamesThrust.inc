//===---APINamesMapThrust.inc-------------------------------------------===//
//
// Copyright (C) 2020 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

// thrust::replace_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::replace_if(thrust::device, device.begin(), device.end(), pred, 0);
      CALL_FACTORY_ENTRY("replace_if",
                         CALL("oneapi::dpl::replace_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
      //Handling case: thrust::replace_if(thrust::seq, host.begin(), host.end(), pred, 0);
      CALL_FACTORY_ENTRY("replace_if",
                         CALL("oneapi::dpl::replace_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::replace_if(device.begin(), device.end(), stencil.begin(), pred, 0);
        CALL_FACTORY_ENTRY("replace_if",
                           CALL(MapNames::getDpctNamespace() + "replace_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::replace_if(host.begin(), host.end(), stencil.begin(), pred, 0);
        CALL_FACTORY_ENTRY("replace_if",
                           CALL(MapNames::getDpctNamespace() + "replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::replace_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred, 0);
          CALL_FACTORY_ENTRY("replace_if",
                             CALL(MapNames::getDpctNamespace() + "replace_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        //Handling case: thrust::replace_if(thrust::seq, host.begin(), host.end(), stencil.begin(), pred, 0);
        CALL_FACTORY_ENTRY("replace_if",
                           CALL(MapNames::getDpctNamespace() + "replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::replace_if(device.begin(), device.end(), pred, 0);
      CALL_FACTORY_ENTRY("replace_if",
                         CALL("oneapi::dpl::replace_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::replace_if(host.begin(), host.end(), pred, 0);
      CALL_FACTORY_ENTRY("replace_if",
                         CALL("oneapi::dpl::replace_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3))))))

// thrust::remove_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), pred);
      CALL_FACTORY_ENTRY("remove_if",
                         CALL("oneapi::dpl::remove_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::remove_if(thrust::seq, host.begin(), host.end(), pred);
      CALL_FACTORY_ENTRY("remove_if",
                         CALL("oneapi::dpl::remove_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3)))),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::remove_if(device.begin(), device.end(), stencil.begin(), pred);
        CALL_FACTORY_ENTRY("remove_if",
                           CALL(MapNames::getDpctNamespace() + "remove_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::remove_if(host.begin(), host.end(), stencil.begin(), pred);
        CALL_FACTORY_ENTRY("remove_if",
                           CALL(MapNames::getDpctNamespace() + "remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred);
        CALL_FACTORY_ENTRY("remove_if",
                           CALL(MapNames::getDpctNamespace() + "remove_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::remove_if(thrust::seq, host.begin(), host.end(), stencil.begin(), pred);
        CALL_FACTORY_ENTRY("remove_if",
                           CALL(MapNames::getDpctNamespace() + "remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4))))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::remove_if(device.begin(), device.end(), pred);
      CALL_FACTORY_ENTRY("remove_if",
                         CALL("oneapi::dpl::remove_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2))),
      //Handling case: thrust::remove_if(host.begin(), host.end(), pred);
      CALL_FACTORY_ENTRY("remove_if",
                         CALL("oneapi::dpl::remove_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2))))))

// thrust::remove_copy_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), result, pred);
      CALL_FACTORY_ENTRY("remove_copy_if",
                         CALL("oneapi::dpl::remove_copy_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
      //Handling case: thrust::remove_copy_if(thrust::seq, host.begin(), host.end(), result, pred);
      CALL_FACTORY_ENTRY("remove_copy_if",
                         CALL("oneapi::dpl::remove_copy_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::remove_copy_if(device.begin(), device.end(), stencil.begin(), result, pred);
        CALL_FACTORY_ENTRY("remove_copy_if",
                           CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::remove_copy_if(host.begin(), host.end(), stencil.begin(), result, pred);
        CALL_FACTORY_ENTRY("remove_copy_if",
                           CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), stencil.begin(), result, pred);
        CALL_FACTORY_ENTRY("remove_copy_if",
                           CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        //Handling case: thrust::remove_copy_if(thrust::seq, host.begin(), host.end(), stencil.begin(), result, pred);
        CALL_FACTORY_ENTRY("remove_copy_if",
                           CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::remove_copy_if(device.begin(), device.end(), result, pred);
      CALL_FACTORY_ENTRY("remove_copy_if",
                         CALL("oneapi::dpl::remove_copy_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::remove_copy_if(host.begin(), host.end(), result, pred);
      CALL_FACTORY_ENTRY("remove_copy_if",
                         CALL("oneapi::dpl::remove_copy_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3))))))

// thrust::not1
CALL_FACTORY_ENTRY("not1", CALL("oneapi::dpl::not1", ARG(0)))

// thrust::any_of
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("any_of",
                         CALL("oneapi::dpl::any_of", 
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3))),
      CALL_FACTORY_ENTRY("any_of", CALL("oneapi::dpl::any_of", ARG("oneapi::dpl::execution::seq"),
                         ARG(1), ARG(2), ARG(3)))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("any_of",
                         CALL("oneapi::dpl::any_of", 
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2))),
      CALL_FACTORY_ENTRY("any_of", CALL("oneapi::dpl::any_of", ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2)))))

// thrust::replace
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("replace",
                         CALL("oneapi::dpl::replace", 
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
      CALL_FACTORY_ENTRY("replace", CALL("oneapi::dpl::replace", ARG("oneapi::dpl::execution::seq"),
                         ARG(1), ARG(2), ARG(3), ARG(4)))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("replace",
                         CALL("oneapi::dpl::replace", 
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3))),
      CALL_FACTORY_ENTRY("replace", CALL("oneapi::dpl::replace", ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2), ARG(3)))))

// thrust::adjacent_difference
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::adjacent_difference(thrust::device, device.begin(), device.end(), result);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::adjacent_difference(thrust::seq, host.begin(), host.end(), result);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::adjacent_difference(device.begin(), device.end(), result, op);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result, op);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::adjacent_difference(thrust::device, device.begin(), device.end(), result, op);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
      //Handling case: thrust::adjacent_difference(thrust::seq, host.begin(), host.end(), result, op);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::adjacent_difference(device.begin(), device.end(), result);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2))),
      //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result);
      CALL_FACTORY_ENTRY("adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2))))))

// thrust::inclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::inclusive_scan(thrust::device, device.begin(), device.end(), result);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::inclusive_scan(thrust::seq, host.begin(), host.end(), result);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::inclusive_scan(device.begin(), device.end(), result, op);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3))),
      //thrust::inclusive_scan(host.begin(), host.end(), result, op);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::inclusive_scan(thrust::device, device.begin(), device.end(), result, op);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
      //Handling case: thrust::inclusive_scan(thrust::seq, host.begin(), host.end(), result, op);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      //Handling case: thrust::inclusive_scan(device.begin(), device.end(), result);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2))),
      //Handling case: thrust::inclusive_scan(host.begin(), host.end(), result);
      CALL_FACTORY_ENTRY("inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2))))))

// thrust::gather
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_gather,
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),d_output.begin());
        CALL_FACTORY_ENTRY("gather",
                          CALL(MapNames::getDpctNamespace() + "gather",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        // Handling case: thrust::gather(thrust::seq, h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
        CALL_FACTORY_ENTRY("gather", CALL(MapNames::getDpctNamespace() + "gather", ARG("oneapi::dpl::execution::seq"),
                          ARG(1), ARG(2), ARG(3), ARG(4))))),
  FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_gather,
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::gather(d_map.begin(), d_map.end(), d_values.begin(), d_output.begin());
        CALL_FACTORY_ENTRY("gather",
                          CALL(MapNames::getDpctNamespace() + "gather",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        // Handling case: thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
        CALL_FACTORY_ENTRY("gather", CALL(MapNames::getDpctNamespace() + "gather", ARG("oneapi::dpl::execution::seq"),
                          ARG(0), ARG(1), ARG(2), ARG(3))))))

// thrust::scatter
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_scatter,
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::scatter(thrust::device, d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
        CALL_FACTORY_ENTRY("scatter",
                          CALL(MapNames::getDpctNamespace() + "scatter",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        // Handling case: thrust::scatter(thrust::seq, h_values.begin(), h_values.end(), h_map.begin(), h_output.begin());
        CALL_FACTORY_ENTRY("scatter", CALL(MapNames::getDpctNamespace() + "scatter", ARG("oneapi::dpl::execution::seq"),
                          ARG(1), ARG(2), ARG(3), ARG(4))))),
  FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_scatter,
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
        // Handling case: thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
        CALL_FACTORY_ENTRY("scatter",
                          CALL(MapNames::getDpctNamespace() + "scatter",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        // Handling case: thrust::scatter(h_values.begin(), h_values.end(), h_map.begin(), h_output.begin());
        CALL_FACTORY_ENTRY("scatter", CALL(MapNames::getDpctNamespace() + "scatter", ARG("oneapi::dpl::execution::seq"),
                          ARG(0), ARG(1), ARG(2), ARG(3))))))



// thrust::unique_by_key_copy
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(),d_output_keys.begin(), d_output_values.begin());
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(1), ARG(2), ARG(3), ARG(4),  ARG(5))),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::seq, h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin());
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                           CALL(MapNames::getDpctNamespace() + "unique_copy",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin(), binary_pred);
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                           CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))))),
  CONDITIONAL_FACTORY_ENTRY(
    CompareArgType(0, 1),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
          CALL_FACTORY_ENTRY("unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(thrust::seq, h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin(), binary_pred);
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                           CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(),d_output_keys.begin(), d_output_values.begin());
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3),ARG(4))),
        // Handling case:
        // *new_last_vec.begin() = thrust::unique_by_key_copy(h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin());
        CALL_FACTORY_ENTRY("unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))))