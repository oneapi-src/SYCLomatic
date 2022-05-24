//===--------------------APINamesCUDNN.inc----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

// Base API
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_engine_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnCreate", DEREF(ARG_WC(0)), false,
                            "create_engine")))

REMOVE_API_FACTORY_ENTRY("cudnnDestroy")

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_engine_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetStream", ARG_WC(0), false, "set_queue",
                            ARG_WC(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_engine_ext,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudnnGetStream", DEREF(ARG_WC(1)),
                            MEMBER_CALL(ARG_WC(0), false, "get_queue"))))

// Memory API
REMOVE_API_FACTORY_ENTRY("cudnnCreateTensorDescriptor")

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetTensor4dDescriptor", ARG_WC(0), false,
                            "set", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4),
                            ARG_WC(5), ARG_WC(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetTensor4dDescriptorEx", ARG_WC(0), false,
                            "set", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4),
                            ARG_WC(5), ARG_WC(6), ARG_WC(7), ARG_WC(8),
                            ARG_WC(9))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetTensorNdDescriptor", ARG_WC(0), false,
                            "set", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetTensorNdDescriptorEx", ARG_WC(0), false,
                            "set", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4))))

REMOVE_API_FACTORY_ENTRY("cudnnDestroyTensorDescriptor")

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnGetTensor4dDescriptor", ARG_WC(0), false,
                            "get", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4),
                            ARG_WC(5), ARG_WC(6), ARG_WC(7), ARG_WC(8),
                            ARG_WC(9))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnGetTensorNdDescriptor", ARG_WC(0), false,
                            "get", ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4),
                            ARG_WC(5))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_memory_desc_ext,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudnnGetTensorSizeInBytes", DEREF(ARG_WC(1)),
                            MEMBER_CALL(ARG_WC(0), false, "get_size"))))

// Simple Operation
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_reorder,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnTransformTensor", ARG_WC(0), false, "reorder",
                            DEREF(ARG_WC(1)), ARG_WC(2), ARG_WC(3),
                            DEREF(ARG_WC(4)), ARG_WC(5), ARG_WC(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_fill,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetTensor", ARG_WC(0), false, "fill",
                            ARG_WC(1), ARG_WC(2), ARG_WC(3))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_scale,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnScaleTensor", ARG_WC(0), false, "scale",
                            DEREF(ARG_WC(3)), ARG_WC(1), ARG_WC(2))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_sum,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnAddTensor", ARG_WC(0), false, "sum",
                            DEREF(ARG_WC(1)), ARG_WC(2), ARG_WC(3),
                            DEREF(ARG_WC(4)), ARG_WC(5), ARG_WC(6))))

// Activation Layer
REMOVE_API_FACTORY_ENTRY("cudnnCreateActivationDescriptor")

REMOVE_API_FACTORY_ENTRY("cudnnDestroyActivationDescriptor")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_activation_desc,
    WARNING_FACTORY_ENTRY("cudnnSetActivationDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnSetActivationDescriptor", ARG_WC(0), false,
                              "set", ARG_WC(1), ARG_WC(3))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_activation_desc,
    WARNING_FACTORY_ENTRY("cudnnGetActivationDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnGetActivationDescriptor", ARG_WC(0), false,
                              "get", ARG_WC(1), ARG_WC(3))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_activation_desc,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetActivationDescriptorSwishBeta", ARG_WC(0),
                            false, "set_beta", ARG_WC(1))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_activation_desc,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudnnGetActivationDescriptorSwishBeta",
                            DEREF(ARG_WC(1)),
                            MEMBER_CALL(ARG_WC(0), false, "get_beta"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_activation_forward,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnActivationForward", ARG_WC(0), false,
                            "activation_forward", ARG_WC(1), DEREF(ARG_WC(2)),
                            ARG_WC(3), ARG_WC(4), DEREF(ARG_WC(5)), ARG_WC(6),
                            ARG_WC(7))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_activation_backward,
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
        "cudnnActivationBackward", ARG_WC(0), false, "activation_backward",
        ARG_WC(1), DEREF(ARG_WC(2)), ARG_WC(3), ARG_WC(4), ARG_WC(5), ARG_WC(6),
        ARG_WC(7), ARG_WC(8), DEREF(ARG_WC(9)), ARG_WC(10), ARG_WC(11))))

// LRN Layer
REMOVE_API_FACTORY_ENTRY("cudnnCreateLRNDescriptor")

REMOVE_API_FACTORY_ENTRY("cudnnDestroyLRNDescriptor")

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_lrn_desc,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSetLRNDescriptor", ARG_WC(0), false, "set",
                            ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_lrn_desc,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnGetLRNDescriptor", ARG_WC(0), false, "get",
                            ARG_WC(1), ARG_WC(2), ARG_WC(3), ARG_WC(4))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_lrn_forward,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnLRNCrossChannelForward", ARG_WC(0), false,
                            "lrn_forward", ARG_WC(1), DEREF(ARG_WC(3)), ARG_WC(4),
                            ARG_WC(5), DEREF(ARG_WC(6)), ARG_WC(7), ARG_WC(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_lrn_backward,
    WARNING_FACTORY_ENTRY("cudnnLRNCrossChannelBackward",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnLRNCrossChannelBackward", ARG_WC(0), false,
                              "lrn_backward", ARG_WC(1), DEREF(ARG_WC(3)), ARG_WC(4),
                              ARG_WC(5), ARG_WC(6), ARG_WC(7), ARG_WC(8),
                              ARG_WC(9), DEREF(ARG_WC(10)), ARG_WC(11), ARG_WC(12))),
                          Diagnostics::PRIMITIVE_WORKSPACE, ARG("lrn_backward"),
                          ARG("lrn_forward")))

// Pooling Layer
REMOVE_API_FACTORY_ENTRY("cudnnCreatePoolingDescriptor")

REMOVE_API_FACTORY_ENTRY("cudnnDestroyPoolingDescriptor")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_pooling_desc,
    WARNING_FACTORY_ENTRY("cudnnSetPooling2dDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnSetPooling2dDescriptor", ARG_WC(0), false,
                              "set", ARG_WC(1), ARG_WC(3), ARG_WC(4), ARG_WC(5),
                              ARG_WC(6), ARG_WC(7), ARG_WC(8))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_pooling_desc,
    WARNING_FACTORY_ENTRY("cudnnSetPoolingNdDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnSetPoolingNdDescriptor", ARG_WC(0), false,
                              "set", ARG_WC(1), ARG_WC(3), ARG_WC(4), ARG_WC(5),
                              ARG_WC(6))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_pooling_desc,
    WARNING_FACTORY_ENTRY("cudnnGetPooling2dDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnGetPooling2dDescriptor", ARG_WC(0), false,
                              "get", ARG_WC(1), ARG_WC(3), ARG_WC(4), ARG_WC(5),
                              ARG_WC(6), ARG_WC(7), ARG_WC(8))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_pooling_desc,
    WARNING_FACTORY_ENTRY("cudnnGetPooling2dDescriptor",
                          ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                              "cudnnGetPoolingNdDescriptor", ARG_WC(0), false,
                              "get", ARG_WC(1), ARG_WC(2), ARG_WC(4), ARG_WC(5),
                              ARG_WC(6), ARG_WC(7))),
                          Diagnostics::API_NOT_MIGRATED,
                          ARG("Nan numbers propagation option")))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_pooling_desc,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnGetPooling2dForwardOutputDim", ARG_WC(0),
                            false, "get_forward_output_dim", ARG_WC(1),
                            ARG_WC(2), ARG_WC(3), ARG_WC(4), ARG_WC(5))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_pooling_desc,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnGetPoolingNdForwardOutputDim", ARG_WC(0),
                            false, "get_forward_output_dim", ARG_WC(1),
                            ARG_WC(2), ARG_WC(3))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_pooling_forward,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnPoolingForward", ARG_WC(0), false,
                            "pooling_forward", ARG_WC(1), DEREF(ARG_WC(2)), ARG_WC(3),
                            ARG_WC(4), DEREF(ARG_WC(5)), ARG_WC(6), ARG_WC(7))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DnnlUtils_pooling_backward,
    WARNING_FACTORY_ENTRY(
        "cudnnPoolingBackward",
        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cudnnPoolingBackward", ARG_WC(0), false, "pooling_backward",
            ARG_WC(1), DEREF(ARG_WC(2)), ARG_WC(3), ARG_WC(4), ARG_WC(5), ARG_WC(6),
            ARG_WC(7), ARG_WC(8), DEREF(ARG_WC(9)), ARG_WC(10), ARG_WC(11))),
        Diagnostics::PRIMITIVE_WORKSPACE, ARG("pooling_backward"),
        ARG("pooling_forward")))

// Softmax Layer
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_softmax_forward,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSoftmaxForward", ARG_WC(0), false,
                            "softmax_forward", ARG_WC(1), ARG_WC(2), DEREF(ARG_WC(3)),
                            ARG_WC(4), ARG_WC(5), DEREF(ARG_WC(6)), ARG_WC(7),
                            ARG_WC(8))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DnnlUtils_softmax_backward,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cudnnSoftmaxBackward", ARG_WC(0), false,
                            "softmax_backward", ARG_WC(1), ARG_WC(2), DEREF(ARG_WC(3)),
                            ARG_WC(4), ARG_WC(5), ARG_WC(6), ARG_WC(7),
                            DEREF(ARG_WC(8)), ARG_WC(9), ARG_WC(10))))
