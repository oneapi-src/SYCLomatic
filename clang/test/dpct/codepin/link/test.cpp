// RUN: %if build_lit %{ icpx -fsycl -c %s -o %T/test.o %}
// RUN: %if build_lit %{ icpx -fsycl -c %S/test2.cpp -o %T/test2.o %}
// RUN: %if build_lit %{ icpx -fsycl  %T/test.o  %T/test2.o %}

#include <dpct/codepin/serialization/basic.hpp>
#include <sycl/sycl.hpp>
void test() {
	dpct::experimental::codepin::detail::demangle_name<sycl::half>();
}

int main() {
	return 0;
}