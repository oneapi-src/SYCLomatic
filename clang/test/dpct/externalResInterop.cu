// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.2
// RUN: dpct --use-experimental-features=bindless_images --format-range=none -out-root %T/extResInterop %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/extResInterop/externalResInterop.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/extResInterop/externalResInterop.dp.cpp -o %T/extResInterop/externalResInterop.dp.o %}

#include <cuda.h>

int main() {
  /// inputs
  int fd;
  void *win_nt_handle;
  const void *obj;
  size_t sizeInBytes;
  // CHECK: sycl::ext::oneapi::experimental::external_mem_handle_type type = sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd;
  // CHECK-NEXT: unsigned int flags = 0;
  cudaExternalMemoryHandleType type = cudaExternalMemoryHandleTypeOpaqueFd;
  unsigned int flags = cudaExternalMemoryDedicated;

  unsigned int numLevels;
  // CHECK: sycl::range<3> extent{0, 0, 0};
  // CHECK-NEXT: auto mip_flags = sycl::ext::oneapi::experimental::image_type::standard;
  // CHECK-NEXT: dpct::image_channel formatDesc;
  cudaExtent extent;
  auto mip_flags = cudaArrayDefault;
  cudaChannelFormatDesc	formatDesc;

  void* devPtr;
  unsigned int flag_0 = 0;
  unsigned long long offset;


  /// types
  // CHECK: sycl::ext::oneapi::experimental::external_mem extMem;
  // CHECK-NEXT: dpct::experimental::image_mem_wrapper_ptr mipmap;
  // CHECK-NEXT: dpct::experimental::external_mem_handle_desc memHandleDesc;
  // CHECK-NEXT: dpct::experimental::external_mem_img_desc mipmappedArrDesc;
  // CHECK-NEXT: dpct::experimental::external_mem_buf_desc bufferDesc;
  cudaExternalMemory_t extMem;
  cudaMipmappedArray_t mipmap;
  cudaExternalMemoryHandleDesc memHandleDesc;
  cudaExternalMemoryMipmappedArrayDesc mipmappedArrDesc;
  cudaExternalMemoryBufferDesc bufferDesc;


  /// setters
#ifdef _WIN32
  // CHECK-WINDOWS: memHandleDesc.set_win32_handle(win_nt_handle);
  // CHECK-WINDOWS-NEXT: memHandleDesc.set_win32_obj_name(obj);
  memHandleDesc.handle.win32.handle = win_nt_handle;
  memHandleDesc.handle.win32.name = obj;
#else
  // CHECK-LINUX: memHandleDesc.set_fd_handle(fd);
  memHandleDesc.handle.fd = fd;
#endif // _WIN32

  // CHECK: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::win32_nt_handle);
  // CHECK-NEXT: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::win32_nt_dx12_resource);
  // CHECK-NEXT: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd);
  // CHECK-NEXT: memHandleDesc.set_handle_type(type);
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  memHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memHandleDesc.type = type;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryHandleTypeOpaqueWin32Kmt is not supported.
  // CHECK-NEXT: */
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
#endif // !NO_BUILD_TEST

  // CHECK: memHandleDesc.set_flags(flags);
  // CHECK-NEXT: memHandleDesc.set_flags(0);
  // CHECK-NEXT: memHandleDesc.set_flags(0);
  memHandleDesc.flags = flags;
  memHandleDesc.flags = 0;
  memHandleDesc.flags = cudaExternalMemoryDedicated;

  // CHECK: memHandleDesc.set_res_size(9);
  // CHECK-NEXT: memHandleDesc.set_res_size(sizeInBytes);
  memHandleDesc.size = 9;
  memHandleDesc.size = sizeInBytes;

  // CHECK: mipmappedArrDesc.set_size(extent);
  mipmappedArrDesc.extent = extent;

  // CHECK: mipmappedArrDesc.set_image_type(mip_flags);
  // CHECK-NEXT: mipmappedArrDesc.set_image_type(sycl::ext::oneapi::experimental::image_type::mipmap);
  mipmappedArrDesc.flags = mip_flags;
  mipmappedArrDesc.flags = cudaArraySurfaceLoadStore;
  #ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaArraySparse is not supported.
  // CHECK-NEXT: */
  mipmappedArrDesc.flags = cudaArraySparse;
#endif // !NO_BUILD_TEST

  // CHECK: mipmappedArrDesc.set_image_channel(formatDesc);
  mipmappedArrDesc.formatDesc = formatDesc;

  // CHECK: mipmappedArrDesc.set_num_levels(numLevels);
  // CHECK-NEXT: mipmappedArrDesc.set_num_levels(9);
  mipmappedArrDesc.numLevels = numLevels;
  mipmappedArrDesc.numLevels = 9;

#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryMipmappedArrayDesc::offset is not supported.
  // CHECK-NEXT: */
  mipmappedArrDesc.offset = offset;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryMipmappedArrayDesc::offset is not supported.
  // CHECK-NEXT: */
  mipmappedArrDesc.offset = 9;
#endif // !NO_BUILD_TEST

  // CHECK: bufferDesc.set_flags(flag_0);
  // CHECK-NEXT: bufferDesc.set_flags(0);
  bufferDesc.flags = flag_0;
  bufferDesc.flags = 0;

  // CHECK: bufferDesc.set_mem_offset(offset);
  // CHECK-NEXT: bufferDesc.set_mem_offset(9);
  bufferDesc.offset = offset;
  bufferDesc.offset = 9;

  // CHECK: bufferDesc.set_res_size(sizeInBytes);
  // CHECK-NEXT: bufferDesc.set_res_size(9);
  bufferDesc.size = sizeInBytes;
  bufferDesc.size = 9;


  /// getters
#ifdef _WIN32
  // CHECK-WINDOWS: win_nt_handle = memHandleDesc.get_win32_handle();
  // CHECK-WINDOWS-NEXT: obj = memHandleDesc.get_win32_obj_name();
  win_nt_handle = memHandleDesc.handle.win32.handle;
  obj = memHandleDesc.handle.win32.name;
#else
  // CHECK-LINUX: fd = memHandleDesc.get_fd_handle();
  fd = memHandleDesc.handle.fd;
#endif // _WIN32
  // CHECK: type = memHandleDesc.get_handle_type();
  // CHECK-NEXT: flags = memHandleDesc.get_flags();
  // CHECK-NEXT: sizeInBytes = memHandleDesc.get_res_size();
  type = memHandleDesc.type;
  flags = memHandleDesc.flags;
  sizeInBytes = memHandleDesc.size;

  // CHECK: extent = mipmappedArrDesc.get_size();
  // CHECK-NEXT: mip_flags = mipmappedArrDesc.get_image_type();
  // CHECK-NEXT: formatDesc = mipmappedArrDesc.get_image_channel();
  // CHECK-NEXT: numLevels = mipmappedArrDesc.get_num_levels();
  extent = mipmappedArrDesc.extent;
  mip_flags = mipmappedArrDesc.flags;
  formatDesc = mipmappedArrDesc.formatDesc;
  numLevels = mipmappedArrDesc.numLevels;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryMipmappedArrayDesc::offset is not supported.
  // CHECK-NEXT: */
  offset = mipmappedArrDesc.offset;
#endif // !NO_BUILD_TEST

  // CHECK: flags = bufferDesc.get_flags();
  // CHECK-NEXT: offset = bufferDesc.get_mem_offset();
  // CHECK-NEXT: sizeInBytes = bufferDesc.get_res_size();
  flags = bufferDesc.flags;
  offset = bufferDesc.offset;
  sizeInBytes = bufferDesc.size;


  /// calls
  // CHECK: dpct::experimental::import_external_memory(&extMem, &memHandleDesc);
  // CHECK-NEXT: mipmap = new dpct::experimental::image_mem_wrapper(extMem, &mipmappedArrDesc);
  // CHECK-NEXT: devPtr = sycl::ext::oneapi::experimental::map_external_linear_memory(extMem, (&bufferDesc)->get_res_size(), (&bufferDesc)->get_mem_offset(), q_ct1);
  // CHECK-NEXT: sycl::ext::oneapi::experimental::release_external_memory(extMem, q_ct1);
  cudaImportExternalMemory(&extMem, &memHandleDesc);
  cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &mipmappedArrDesc);
  cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
  cudaDestroyExternalMemory(extMem);

  return 0;
}
