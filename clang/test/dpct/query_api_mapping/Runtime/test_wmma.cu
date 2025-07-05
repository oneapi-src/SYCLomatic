// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvcuda::wmma::fill_fragment | FileCheck %s -check-prefix=NVCUDA_WMMA_FILL_FRAGMENT
// NVCUDA_WMMA_FILL_FRAGMENT: CUDA API:
// NVCUDA_WMMA_FILL_FRAGMENT-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_FILL_FRAGMENT-NEXT:    nvcuda::wmma::fill_fragment(acc_frag, val /*const T&*/);
// NVCUDA_WMMA_FILL_FRAGMENT-NEXT: Is migrated to (with the option --use-experimental-features=matrix):
// NVCUDA_WMMA_FILL_FRAGMENT-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_FILL_FRAGMENT-NEXT:    sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), val);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvcuda::wmma::load_matrix_sync | FileCheck %s -check-prefix=NVCUDA_WMMA_LOAD_MATRIX_SYNC
// NVCUDA_WMMA_LOAD_MATRIX_SYNC: CUDA API:
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:                           nvcuda::wmma::row_major>
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:        a_frag;
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:    nvcuda::wmma::load_matrix_sync(a_frag, a + col + row * lda /*const T **/,
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:                                   lda /*unsigned*/);
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=matrix):
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::a, 16, 16, 16, sycl::half, dpct::experimental::matrix::row_major>
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:        a_frag;
// NVCUDA_WMMA_LOAD_MATRIX_SYNC-NEXT:    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sycl::ext::oneapi::this_work_item::get_sub_group(), a_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, typename std::remove_pointer<decltype(a + col + row * lda)>::type>(a + col + row * lda), lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvcuda::wmma::store_matrix_sync | FileCheck %s -check-prefix=NVCUDA_WMMA_STORE_MATRIX_SYNC
// NVCUDA_WMMA_STORE_MATRIX_SYNC: CUDA API:
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    nvcuda::wmma::store_matrix_sync(
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:        c + col + row * ldc /*const T **/, acc_frag, ldc /*unsigned*/,
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:        nvcuda::wmma::mem_col_major /*nvcuda::wmma::layout_t*/);
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    nvcuda::wmma::store_matrix_sync(
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:        c + row + col * ldc /*const T **/, acc_frag, ldc /*unsigned*/,
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:        nvcuda::wmma::mem_row_major /*nvcuda::wmma::layout_t*/);
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=matrix):
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, typename std::remove_pointer<decltype(c + col + row * ldc)>::type>(c + col + row * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::col_major);
// NVCUDA_WMMA_STORE_MATRIX_SYNC-NEXT:    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, typename std::remove_pointer<decltype(c + row + col * ldc)>::type>(c + row + col * ldc), ldc, sycl::ext::oneapi::experimental::matrix::layout::row_major);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvcuda::wmma::mma_sync | FileCheck %s -check-prefix=NVCUDA_WMMA_MMA_SYNC
// NVCUDA_WMMA_MMA_SYNC: CUDA API:
// NVCUDA_WMMA_MMA_SYNC-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
// NVCUDA_WMMA_MMA_SYNC-NEXT:                           nvcuda::wmma::row_major>
// NVCUDA_WMMA_MMA_SYNC-NEXT:        a_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
// NVCUDA_WMMA_MMA_SYNC-NEXT:                           nvcuda::wmma::col_major>
// NVCUDA_WMMA_MMA_SYNC-NEXT:        b_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
// NVCUDA_WMMA_MMA_SYNC-NEXT: Is migrated to (with the option --use-experimental-features=matrix):
// NVCUDA_WMMA_MMA_SYNC-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::a, 16, 16, 16, sycl::half, dpct::experimental::matrix::row_major>
// NVCUDA_WMMA_MMA_SYNC-NEXT:        a_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::b, 16, 16, 16, sycl::half, dpct::experimental::matrix::col_major>
// NVCUDA_WMMA_MMA_SYNC-NEXT:        b_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    dpct::experimental::matrix::joint_matrix<dpct::experimental::matrix::accumulator, 16, 16, 16, float> acc_frag;
// NVCUDA_WMMA_MMA_SYNC-NEXT:    sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sycl::ext::oneapi::this_work_item::get_sub_group(), acc_frag.get(), a_frag.get(), b_frag.get(), acc_frag.get());
