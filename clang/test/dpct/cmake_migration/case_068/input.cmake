## execute_process is commented in the CMakeLists.txt file
set(AOT_GENERATE_COMMAND
    ${Python3_EXECUTABLE} -m aot_build_utils.generate --path
    ${GENERATED_SOURCE_DIR} --head_dims ${HEAD_DIMS} --pos_encoding_modes
    ${POS_ENCODING_MODES} --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES} --enable_f16 ${FLASHINFER_ENABLE_F16}
    --enable_bf16 ${FLASHINFER_ENABLE_BF16} --enable_fp8_e4m3
    ${FLASHINFER_ENABLE_FP8_E4M3} --enable_fp8_e5m2
    ${FLASHINFER_ENABLE_FP8_E5M2})

set(AOT_GENERATE_DISPATCH_INC_COMMAND
    ${Python3_EXECUTABLE} -m aot_build_utils.generate_dispatch_inc --path
    "${GENERATED_SOURCE_DIR}/dispatch.inc" --head_dims ${HEAD_DIMS}
    --head_dims_sm90 ${HEAD_DIMS_SM90} --pos_encoding_modes
    ${POS_ENCODING_MODES} --use_fp16_qk_reductions ${USE_FP16_QK_REDUCTIONS}
    --mask_modes ${MASK_MODES})

execute_process(COMMAND ${AOT_GENERATE_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
execute_process(COMMAND ${AOT_GENERATE_DISPATCH_INC_COMMAND}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
