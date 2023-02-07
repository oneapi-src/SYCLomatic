// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DplExtrasIterators/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasIterators/api_test6_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasIterators/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasIterators/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasIterators/api_test6_out

// CHECK: 5
// TEST_FEATURE: DplExtrasIterators_zip_iterator

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

template<typename int_iterator>
void foo() {
  typedef thrust::tuple<int_iterator, int_iterator> iterator_tuple;
  typedef thrust::zip_iterator<iterator_tuple> int_zip_iterator;
}

int main() {
  return 0;
}
