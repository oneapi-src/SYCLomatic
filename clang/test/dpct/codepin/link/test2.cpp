// RUN: echo 0
#include <dpct/codepin/serialization/basic.hpp>
#include <sycl/sycl.hpp>

void test2() {
	dpct::experimental::codepin::detail::demangle_name<sycl::half>();
}
