target_compile_options(tiny-cuda-nn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

target_compile_options(
  foo
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
          >)

target_compile_options(foo PUBLIC $<$<COMPILE_LANGUAGE:CUDA,NVIDIA>: -Wno-unused-but-set-variable>)
target_compile_options(faiss_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all --expt-extended-lambda --expt-relaxed-constexpr>)
