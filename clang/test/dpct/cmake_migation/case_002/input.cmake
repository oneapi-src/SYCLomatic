project(foo-bar LANGUAGES CXX CUDA)

project(foo2 CUDA CXX)

project(
	foo
	VERSION 1.7
	DESCRIPTION cuda project
	LANGUAGES CXX CUDA
)

project(
  foo
  VERSION 1.1.0
  LANGUAGES)