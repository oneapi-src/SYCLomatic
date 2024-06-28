set(CUDA_HAS_FP16 TRUE)
file(WRITE ${output_file} "SET(CUDA_HAS_FP16\n TRUE)\n\n")
file(WRITE ${output_file} "SET(CUDA_HAS_FP16 TRUE)\n\n")

install(
  EXPORT FooTargets
  FILE FooTargets.cmake
  NAMESPACE Foo::
  DESTINATION lib/cmake/Foo)
