#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

void reduceBlock(double *sdata, const sycl::group<3> &cta,
                 const sycl::nd_item<3> &item_ct1) {
  const unsigned int tid = item_ct1.get_local_linear_id();
  sycl::sub_group tile32 = item_ct1.get_sub_group();

  /*
  DPCT1007:1: Migration of reduce is not supported.
  */
  cg::reduce(tile32, sdata[tid], cg::plus<double>());
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  */
  item_ct1.barrier();
}
