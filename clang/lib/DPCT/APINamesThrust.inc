//===--------------- APINamesThrust.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// thrust::replace_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace_if(policy, ptr, ptr, stc, pred, val);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::replace_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::replace_if",
                            CALL(MapNames::getDpctNamespace() + "replace_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4), ARG(5)))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL(MapNames::getDpctNamespace() + "replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    //Handling case: thrust::replace_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred, 0);
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
        CALL_FACTORY_ENTRY("thrust::replace_if",
                           CALL(MapNames::getDpctNamespace() + "replace_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::replace_if(ptr, ptr, pred, val);
      IFELSE_FACTORY_ENTRY(
        "thrust::replace_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::replace_if",
                            CALL("oneapi::dpl::replace_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  ARG(2), ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                           CALL("oneapi::dpl::replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::replace_if(device.begin(), device.end(), pred, 0);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::replace_if(policy, ptr, ptr, pred, val);
        IFELSE_FACTORY_ENTRY(
          "thrust::replace_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL("oneapi::dpl::replace_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3), ARG(4)))),
          CALL_FACTORY_ENTRY("thrust::replace_if",
                             CALL("oneapi::dpl::replace_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3), ARG(4)))),
        //Handling case: thrust::replace_if(ptr, ptr, stc, pred, val);
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
          IFELSE_FACTORY_ENTRY(
            "thrust::replace_if",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::replace_if",
                                CALL(MapNames::getDpctNamespace() + "replace_if",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      ARG(3), ARG(4)))),
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::replace_if(policy, device.begin(), device.end(), pred, 0);
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::replace_if(device.begin(), device.end(), stencil.begin(), pred, 0);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))))
    )
  )
)

// thrust::remove_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::remove_if(policy, ptr, ptr, stc, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4)))),
        CALL_FACTORY_ENTRY("thrust::remove_if",
                          CALL(MapNames::getDpctNamespace() + "remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4))))),
    //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      CALL_FACTORY_ENTRY("thrust::remove_if",
                        CALL(MapNames::getDpctNamespace() + "remove_if",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::remove_if(ptr, ptr, pred);
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL("oneapi::dpl::remove_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    ARG(2)))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::remove_if(device.begin(), device.end(), pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::remove_if(policy, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_if",
                            CALL("oneapi::dpl::remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::remove_if",
                           CALL("oneapi::dpl::remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::remove_if(ptr, ptr, stc, pred);
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                            CALL(MapNames::getDpctNamespace() + "remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), pred);
        CALL_FACTORY_ENTRY("thrust::remove_if",
                          CALL("oneapi::dpl::remove_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::remove_if(device.begin(), device.end(), stencil.begin(), pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                   CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                   ARG(0), ARG(1), ARG(2), ARG(3)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                   ARG("oneapi::dpl::execution::seq"),
                                   ARG(0), ARG(1), ARG(2), ARG(3))))))
    )
  )
)

// thrust::remove_copy_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::remove_copy_if(policy, ptr, ptr, stc, ptr, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_copy_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                            CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                  ARG(5)))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                          CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), stencil.begin(), result, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                        CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::remove_copy_if(ptr, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_copy_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                            CALL("oneapi::dpl::remove_copy_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::remove_copy_if(device.begin(), device.end(), result, pred);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_copy_if(policy, ptr, ptr, ptr, pred);
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_copy_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL("oneapi::dpl::remove_copy_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                    ARG(4)))),
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                             CALL("oneapi::dpl::remove_copy_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3), ARG(4)))),
        //Handling case: thrust::remove_copy_if(ptr, ptr, stc, ptr, pred);
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
          IFELSE_FACTORY_ENTRY(
            "thrust::remove_copy_if",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                                CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(3)),
                                      ARG(4)))),
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), result, pred);
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                          CALL("oneapi::dpl::remove_copy_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::remove_copy_if(device.begin(), device.end(), stencil.begin(), result, pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                   CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                   ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                   ARG("oneapi::dpl::execution::seq"),
                                   ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))))
    )
  )
)

// thrust::not1
CALL_FACTORY_ENTRY("thrust::not1", CALL("oneapi::dpl::not1", ARG(0)))

// thrust::any_of
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    IFELSE_FACTORY_ENTRY(
      "thrust::any_of",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::any_of", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::any_of",
                            CALL("oneapi::dpl::any_of",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                ARG(3)))),
      CALL_FACTORY_ENTRY("thrust::any_of",
                          CALL("oneapi::dpl::any_of",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CALL_FACTORY_ENTRY("thrust::any_of",
                      CALL("oneapi::dpl::any_of", 
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    IFELSE_FACTORY_ENTRY(
      "thrust::any_of",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::any_of", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::any_of",
                            CALL("oneapi::dpl::any_of",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                ARG(2)))),
      CALL_FACTORY_ENTRY("thrust::any_of",
                          CALL("oneapi::dpl::any_of",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::any_of", CALL("oneapi::dpl::any_of",
                         CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                         ARG(0), ARG(1), ARG(2))),
      CALL_FACTORY_ENTRY("thrust::any_of", CALL("oneapi::dpl::any_of", ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2))))
  )
)

// thrust::replace
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace(policy, ptr, ptr, val, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::replace",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::replace", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::replace",
                            CALL("oneapi::dpl::replace",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                ARG(3), ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::replace",
                          CALL("oneapi::dpl::replace",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CALL_FACTORY_ENTRY("thrust::replace",
                       CALL("oneapi::dpl::replace", 
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace(ptr, ptr, val, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::replace",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::replace", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::replace",
                            CALL("oneapi::dpl::replace",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                ARG(2), ARG(3)))),
      CALL_FACTORY_ENTRY("thrust::replace",
                          CALL("oneapi::dpl::replace",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::replace", CALL("oneapi::dpl::replace",
                         CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                         ARG(0), ARG(1), ARG(2), ARG(3))),
      CALL_FACTORY_ENTRY("thrust::replace", CALL("oneapi::dpl::replace",
                         ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2), ARG(3))))
  )
)

// thrust::adjacent_difference
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::adjacent_difference(policy, ptr, ptr, ptr, op);
    IFELSE_FACTORY_ENTRY(
      "thrust::adjacent_difference",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    //Handling case: thrust::adjacent_difference(policy, device.begin(), device.end(), result, op);
    CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                      CALL("oneapi::dpl::adjacent_difference",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::adjacent_difference(ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::adjacent_difference",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                           CALL("oneapi::dpl::adjacent_difference",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                               CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                               ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                               ARG("oneapi::dpl::execution::seq"),
                               ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::adjacent_difference(policy, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::adjacent_difference",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                              CALL("oneapi::dpl::adjacent_difference",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                             CALL("oneapi::dpl::adjacent_difference",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3)))),
        //Handling case: thrust::adjacent_difference(ptr, ptr, ptr, op);
        IFELSE_FACTORY_ENTRY(
          "thrust::adjacent_difference",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                              CALL("oneapi::dpl::adjacent_difference",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                             CALL("oneapi::dpl::adjacent_difference",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::adjacent_difference(policy, device.begin(), device.end(), result);
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result, op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                 ARG(0), ARG(1), ARG(2), ARG(3))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                 ARG("oneapi::dpl::execution::seq"),
                                 ARG(0), ARG(1), ARG(2), ARG(3)))))
    )
  )
)

// thrust::inclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::inclusive_scan(policy, ptr, ptr, ptr, op);
    IFELSE_FACTORY_ENTRY(
      "thrust::inclusive_scan",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                          CALL("oneapi::dpl::inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    //Handling case: thrust::inclusive_scan(policy, device.begin(), device.end(), result, op);
    CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                      CALL("oneapi::dpl::inclusive_scan",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::inclusive_scan(ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::inclusive_scan",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                            CALL("oneapi::dpl::inclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::inclusive_scan(host.begin(), host.end(), result);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::inclusive_scan(policy, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::inclusive_scan",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                              CALL("oneapi::dpl::inclusive_scan",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3)))),
        //Handling case: thrust::inclusive_scan(ptr, ptr, ptr, op);
        IFELSE_FACTORY_ENTRY(
          "thrust::inclusive_scan",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                              CALL("oneapi::dpl::inclusive_scan",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::inclusive_scan(policy, device.begin(), device.end(), result);
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                          CALL("oneapi::dpl::inclusive_scan",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::inclusive_scan(host.begin(), host.end(), result, op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2), ARG(3))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3)))))
    )
  )
)

// thrust::gather
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_gather,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::gather(policy, ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::gather",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::gather",
                              CALL(MapNames::getDpctNamespace() + "gather",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
        CALL_FACTORY_ENTRY("thrust::gather",
                            CALL(MapNames::getDpctNamespace() + "gather",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),d_output.begin());
      CALL_FACTORY_ENTRY("thrust::gather",
                        CALL(MapNames::getDpctNamespace() + "gather",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::gather(ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::gather",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::gather",
                              CALL(MapNames::getDpctNamespace() + "gather",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
        CALL_FACTORY_ENTRY("thrust::gather",
                            CALL(MapNames::getDpctNamespace() + "gather",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      // Handling case: thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "gather",
                           CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                           ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "gather", ARG("oneapi::dpl::execution::seq"),
                          ARG(0), ARG(1), ARG(2), ARG(3))))
    )
  )
)

// thrust::scatter
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_scatter,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::scatter(policy, ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::scatter",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::scatter",
                              CALL(MapNames::getDpctNamespace() + "scatter",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
        CALL_FACTORY_ENTRY("thrust::scatter",
                            CALL(MapNames::getDpctNamespace() + "scatter",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::scatter(policy, d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
      CALL_FACTORY_ENTRY("thrust::scatter",
                        CALL(MapNames::getDpctNamespace() + "scatter",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::scatter(ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::scatter",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::scatter",
                              CALL(MapNames::getDpctNamespace() + "scatter",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
        CALL_FACTORY_ENTRY("thrust::scatter",
                            CALL(MapNames::getDpctNamespace() + "scatter",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      // Handling case: thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "scatter",
                           CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                           ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "scatter", ARG("oneapi::dpl::execution::seq"),
                           ARG(0), ARG(1), ARG(2), ARG(3))))
    )
  )
)

// thrust::unique_by_key_copy
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(7),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::unique_by_key_copy(policy, ptr, ptr, ptr, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::unique_by_key_copy",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                            CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(5)), ARG(5)),
                                  ARG(6)))),
        CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))),
      // Handling case: thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
      CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                        CALL(MapNames::getDpctNamespace() + "unique_copy",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(5),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        //Handling case: thrust::unique_by_key_copy(ptr, ptr, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::unique_by_key_copy",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                            CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          // Handling case: thrust::unique_by_key_copy(h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))
      ),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          //Handling case: thrust::unique_by_key_copy(policy, ptr, ptr, ptr, ptr, ptr);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key_copy",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                                CALL(MapNames::getDpctNamespace() + "unique_copy",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(5)), ARG(5))))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))),
          //Handling case: thrust::unique_by_key_copy(ptr, ptr, ptr, ptr, ptr, pred);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key_copy",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                                CALL(MapNames::getDpctNamespace() + "unique_copy",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                      ARG(5)))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          // Handling case: thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(),d_output_keys.begin(), d_output_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  makeMappedThrustPolicyEnum(0),
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            // Handling case: thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                               CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                               CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))))
      )
    )
  )
)

// thrust::transform_reduce
CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(0, "thrust::device_ptr"),
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
        CheckArgType(0, "thrust::device"),
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
