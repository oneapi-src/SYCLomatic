/******************************************************************************
*
* Copyright 2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

#ifndef __DPCT_NUMERIC_H
#define __DPCT_NUMERIC_H


namespace dpct {

template <typename Policy, typename InputIt1, typename InputIt2, typename T>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename T,
          typename BinaryOperation1, typename BinaryOperation2>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init, BinaryOperation1 op1,
                BinaryOperation2 op2) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init, op1, op2);
}

} // end namespace dpct

#endif //__DPCT_NUMERIC_H