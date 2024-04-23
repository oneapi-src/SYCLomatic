project(foo-bar LANGUAGES CXX CUDA)

project(foo2 CUDA CXX)

project(foo
VERSION 1.7
DESCRIPTION "cuda test"
LANGUAGES CXX CUDA
)

project(foo
VERSION 1.1.0
LANGUAGES)

project(foo)

project(
	tiny-cuda-nn
	VERSION 1.7
	DESCRIPTION "Lightning fast & tiny C++/CUDA neural network framework"
	LANGUAGES CXX CUDA
)
