// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/fix_alias_bug %S/fix_alias_bug.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/fix_alias_bug/fix_alias_bug.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/type/fix_alias_bug/fix_alias_bug.dp.cpp -o %T/type/fix_alias_bug/fix_alias_bug.dp.o %}

#include <cub/cub.cuh>
#include <cuda.h>
template <bool> struct warp_reduce;

template <> struct warp_reduce<true> {
  template <typename T, typename reducer_t>
  inline T operator()(const T &value_, bool all, const reducer_t &r) {
    // CHECK: using warp_reduce_t = sycl::sub_group;

    using warp_reduce_t = cub::WarpReduce<T, 10>;

    typename warp_reduce_t::TempStorage dummy_storage;
    // CHECK: warp_reduce_t warp_reduce(sycl::ext::oneapi::this_work_item::get_sub_group());

    warp_reduce_t warp_reduce(dummy_storage);
  }
};

template <> struct warp_reduce<false> {
  template <typename T, typename reducer_t>
  inline T operator()(const T &value_, bool all, const reducer_t &r) {
    // CHECK: using warp_reduce_t = sycl::sub_group;

    using warp_reduce_t = cub::WarpReduce<T, 10>;

    typename warp_reduce_t::TempStorage dummy_storage;
    // CHECK: warp_reduce_t warp_reduce(sycl::ext::oneapi::this_work_item::get_sub_group());
    warp_reduce_t warp_reduce(dummy_storage);
  }
};
int main() {
  int a, b;
  warp_reduce<true>()(a, true, b);
}
