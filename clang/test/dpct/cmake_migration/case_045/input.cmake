# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "-use_fast_math")
