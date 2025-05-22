// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include" > %T/auto_deduce_output.txt 2>&1 || true
// RUN: grep "dpct internal error" %T/auto_deduce_output.txt | wc -l > %T/wc_auto_deduce_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file %T/wc_auto_deduce_output.txt
// RUN: rm -rf %T

// CHECK: 0

__device__ void test_auto() {

  auto tid = get_tid();
}

 namespace test {

 enum norm_type_ {
   NORM1,
   NORM2,
   ABS_MAX,
   ABS_MIN
 };

 template <typename T, bool fixed, int nColor, int... N>
 double norm(const GaugeField &u, int d, norm_type_ type, IntList<nColor, N...>) {
   double norm_ = 0.0;
     if constexpr (sizeof...(N) > 0) {
       norm<T, fixed>(u, d, type, IntList<N...>());
     }
   return norm_;
 }

 } // namespace test