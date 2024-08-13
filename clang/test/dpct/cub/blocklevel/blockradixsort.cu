// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -format-range=none -in-root %S -out-root %T/blocklevel/blockradixsort %S/blockradixsort.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockradixsort/blockradixsort.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockradixsort/blockradixsort.dp.cpp -o %T/blocklevel/blockradixsort/blockradixsort.dp.o %}

#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
__device__ void LoadDirectBlocked(int linear_tid, InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
  }
}

template <typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
__device__ void StoreDirectBlocked(int linear_tid, OutputIteratorT block_itr,
                                   T (&items)[ITEMS_PER_THREAD]) {
  OutputIteratorT thread_itr = block_itr + (linear_tid * ITEMS_PER_THREAD);
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_itr[ITEM] = items[ITEM];
  }
}

__global__ void Sort(int *data) {
  // CHECK: using BlockRadixSort = dpct::group::radix_sort<int, 4>;
  // CHECK-NOT: __shared__ typename BlockRadixSort::TempStorage temp_storage;
  using BlockRadixSort = cub::BlockRadixSort<int, 128, 4>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  int thread_keys[4];
  LoadDirectBlocked(threadIdx.x, data, thread_keys);
  // CHECK: BlockRadixSort(temp_storage).sort(item_ct1, thread_keys);
  BlockRadixSort(temp_storage).Sort(thread_keys);
  StoreDirectBlocked(threadIdx.x, data, thread_keys);
}

__global__ void SortBit(int *data) {
  // CHECK: using BlockRadixSort = dpct::group::radix_sort<int, 4>;
  // CHECK-NOT: __shared__ typename BlockRadixSort::TempStorage temp_storage;
  using BlockRadixSort = cub::BlockRadixSort<int, 128, 4>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  int thread_keys[4];
  LoadDirectBlocked(threadIdx.x, data, thread_keys);
  // CHECK: BlockRadixSort(temp_storage).sort(item_ct1, thread_keys, 4, 16);
  BlockRadixSort(temp_storage).Sort(thread_keys, 4, 16);
  StoreDirectBlocked(threadIdx.x, data, thread_keys);
}

__global__ void SortDescending(int *data) {
  // CHECK: using BlockRadixSort = dpct::group::radix_sort<int, 4>;
  // CHECK-NOT: __shared__ typename BlockRadixSort::TempStorage temp_storage;
  using BlockRadixSort = cub::BlockRadixSort<int, 128, 4>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  int thread_keys[4];
  LoadDirectBlocked(threadIdx.x, data, thread_keys);
  // CHECK: BlockRadixSort(temp_storage).sort_descending(item_ct1, thread_keys);
  BlockRadixSort(temp_storage).SortDescending(thread_keys);
  StoreDirectBlocked(threadIdx.x, data, thread_keys);
}

__global__ void SortDescendingBit(int *data) {
  // CHECK: using BlockRadixSort = dpct::group::radix_sort<int, 4>;
  // CHECK-NOT: __shared__ typename BlockRadixSort::TempStorage temp_storage;
  using BlockRadixSort = cub::BlockRadixSort<int, 128, 4>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  int thread_keys[4];
  LoadDirectBlocked(threadIdx.x, data, thread_keys);
  // CHECK: BlockRadixSort(temp_storage).sort_descending(item_ct1, thread_keys, 4, 16);
  BlockRadixSort(temp_storage).SortDescending(thread_keys, 4, 16);
  StoreDirectBlocked(threadIdx.x, data, thread_keys);
}

bool test_sort() {
  int data[512] = {0}, *d_data = nullptr;
  cudaMalloc(&d_data, sizeof(int) * 512);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::radix_sort<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         Sort(d_data, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  Sort<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(data, d_data, sizeof(data), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  for (int i = 0; i < 512; ++i)
    if (data[i] != i) {
      printf("test_sort failed\n");
      return false;
    }
  printf("test_sort pass\n");
  return true;
}

bool test_sort_bit() {
  int data[512] = {0}, *d_data = nullptr;
  int expected[512] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
      45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
      60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
      75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
      90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
      165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
      195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
      240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
      255, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258,
      257, 256, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
      274, 273, 272, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294, 293, 292,
      291, 290, 289, 288, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309,
      308, 307, 306, 305, 304, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326,
      325, 324, 323, 322, 321, 320, 351, 350, 349, 348, 347, 346, 345, 344, 343,
      342, 341, 340, 339, 338, 337, 336, 367, 366, 365, 364, 363, 362, 361, 360,
      359, 358, 357, 356, 355, 354, 353, 352, 383, 382, 381, 380, 379, 378, 377,
      376, 375, 374, 373, 372, 371, 370, 369, 368, 399, 398, 397, 396, 395, 394,
      393, 392, 391, 390, 389, 388, 387, 386, 385, 384, 415, 414, 413, 412, 411,
      410, 409, 408, 407, 406, 405, 404, 403, 402, 401, 400, 431, 430, 429, 428,
      427, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 447, 446, 445,
      444, 443, 442, 441, 440, 439, 438, 437, 436, 435, 434, 433, 432, 463, 462,
      461, 460, 459, 458, 457, 456, 455, 454, 453, 452, 451, 450, 449, 448, 479,
      478, 477, 476, 475, 474, 473, 472, 471, 470, 469, 468, 467, 466, 465, 464,
      495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 483, 482, 481,
      480, 511, 510, 509, 508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498,
      497, 496};
  cudaMalloc(&d_data, sizeof(int) * 512);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::radix_sort<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         SortBit(d_data, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  SortBit<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(data, d_data, sizeof(data), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_bit failed\n");
      return false;
    }
  printf("test_sort_bit pass\n");
  return true;
}

bool test_sort_descending() {
  int data[512] = {0}, *d_data = nullptr;
  cudaMalloc(&d_data, sizeof(int) * 512);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::radix_sort<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         SortDescending(d_data, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  SortDescending<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(data, d_data, sizeof(data), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  for (int i = 0; i < 512; ++i)
    if (data[i] != 511 - i) {
      printf("test_sort_descending failed\n");
      return false;
    }
  printf("test_sort_descending pass\n");
  return true;
}

bool test_sort_descending_bit() {
  int data[512] = {0}, *d_data = nullptr;
  int expected[512] = {
      511, 510, 509, 508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498, 497,
      496, 495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 483, 482,
      481, 480, 479, 478, 477, 476, 475, 474, 473, 472, 471, 470, 469, 468, 467,
      466, 465, 464, 463, 462, 461, 460, 459, 458, 457, 456, 455, 454, 453, 452,
      451, 450, 449, 448, 447, 446, 445, 444, 443, 442, 441, 440, 439, 438, 437,
      436, 435, 434, 433, 432, 431, 430, 429, 428, 427, 426, 425, 424, 423, 422,
      421, 420, 419, 418, 417, 416, 415, 414, 413, 412, 411, 410, 409, 408, 407,
      406, 405, 404, 403, 402, 401, 400, 399, 398, 397, 396, 395, 394, 393, 392,
      391, 390, 389, 388, 387, 386, 385, 384, 383, 382, 381, 380, 379, 378, 377,
      376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365, 364, 363, 362,
      361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351, 350, 349, 348, 347,
      346, 345, 344, 343, 342, 341, 340, 339, 338, 337, 336, 335, 334, 333, 332,
      331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317,
      316, 315, 314, 313, 312, 311, 310, 309, 308, 307, 306, 305, 304, 303, 302,
      301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289, 288, 287,
      286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275, 274, 273, 272,
      271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257,
      256, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
      254, 255, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
      237, 238, 239, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
      220, 221, 222, 223, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
      203, 204, 205, 206, 207, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
      186, 187, 188, 189, 190, 191, 160, 161, 162, 163, 164, 165, 166, 167, 168,
      169, 170, 171, 172, 173, 174, 175, 144, 145, 146, 147, 148, 149, 150, 151,
      152, 153, 154, 155, 156, 157, 158, 159, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117,
      118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 96, 97, 98, 99, 100,
      101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 80, 81, 82, 83,
      84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64, 65, 66,
      67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 48, 49,
      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 32,
      33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
      31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
      14, 15};
  cudaMalloc(&d_data, sizeof(int) * 512);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::radix_sort<int, 4>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         SortDescendingBit(d_data, item_ct1, &temp_storage_acc[0]);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  SortDescendingBit<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(data, d_data, sizeof(data), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_descending_bit failed\n");
      return false;
    }
  printf("test_sort_descending_bit pass\n");
  return true;
}

int main() { return !(test_sort() && test_sort_bit() && test_sort_descending() && test_sort_descending_bit()); }
