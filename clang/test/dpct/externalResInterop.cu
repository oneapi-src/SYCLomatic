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
  // CHECK: sycl::ext::oneapi::experimental::external_mem_handle_type memHandleType = sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd;
  // CHECK-NEXT: unsigned int memHandleFlags = 0;
  cudaExternalMemoryHandleType memHandleType = cudaExternalMemoryHandleTypeOpaqueFd;
  unsigned int memHandleFlags = cudaExternalMemoryDedicated;

  // CHECK: sycl::ext::oneapi::experimental::external_semaphore_handle_type semHandleType = sycl::ext::oneapi::experimental::external_semaphore_handle_type::opaque_fd;
  cudaExternalSemaphoreHandleType semHandleType = cudaExternalSemaphoreHandleTypeOpaqueFd;
  unsigned int semHandleFlags = 0;

  unsigned int numLevels;
  // CHECK: sycl::range<3> extent{0, 0, 0};
  // CHECK-NEXT: auto mipFlags = sycl::ext::oneapi::experimental::image_type::standard;
  // CHECK-NEXT: dpct::image_channel formatDesc;
  cudaExtent extent;
  auto mipFlags = cudaArrayDefault;
  cudaChannelFormatDesc	formatDesc;

  void* devPtr;
  unsigned int buffFlags = 0;
  unsigned long long offset;

  unsigned long long value;
  void *fence;
  unsigned long long nvSciSync_reserved;
  unsigned long long key;
  unsigned long long timeoutMs;
  // CHECK: unsigned int semSignalParamFlags = 0;
  // CHECK-NEXT: unsigned int semWaitParamFlags = 0;
  unsigned int semSignalParamFlags = cudaExternalSemaphoreSignalSkipNvSciBufMemSync;
  unsigned int semWaitParamFlags = cudaExternalSemaphoreWaitSkipNvSciBufMemSync;

  unsigned int numExtSems = 2;
  cudaStream_t stream;


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

  // CHECK: dpct::experimental::external_sem_wrapper_ptr extSem;
  // CHECK-NEXT: dpct::experimental::external_sem_wrapper_ptr *extSemArr;
  // CHECK-NEXT: dpct::experimental::external_sem_handle_desc semHandleDesc;
  // CHECK-NEXT: dpct::experimental::external_sem_params signalParams;
  // CHECK-NEXT: dpct::experimental::external_sem_params *signalParamsArr;
  // CHECK-NEXT: dpct::experimental::external_sem_params waitParams;
  // CHECK-NEXT: dpct::experimental::external_sem_params *waitParamsArr;
  cudaExternalSemaphore_t extSem;
  cudaExternalSemaphore_t *extSemArr;
  cudaExternalSemaphoreHandleDesc semHandleDesc;
  cudaExternalSemaphoreSignalParams signalParams;
  cudaExternalSemaphoreSignalParams *signalParamsArr;
  cudaExternalSemaphoreWaitParams waitParams;
  cudaExternalSemaphoreWaitParams *waitParamsArr;


  /// setters
#ifdef _WIN32
  // CHECK-WINDOWS: memHandleDesc.set_win32_handle(win_nt_handle);
  // CHECK-WINDOWS-NEXT: memHandleDesc.set_win32_obj_name(obj);
  memHandleDesc.handle.win32.handle = win_nt_handle;
  memHandleDesc.handle.win32.name = obj;

  // CHECK-WINDOWS: semHandleDesc.set_win32_handle(win_nt_handle);
  // CHECK-WINDOWS-NEXT: semHandleDesc.set_win32_obj_name(obj);
  semHandleDesc.handle.win32.handle = win_nt_handle;
  semHandleDesc.handle.win32.name = obj;
#else
  // CHECK-LINUX: memHandleDesc.set_fd_handle(fd);
  memHandleDesc.handle.fd = fd;

  // CHECK-LINUX: semHandleDesc.set_fd_handle(fd);
  semHandleDesc.handle.fd = fd;
#endif // _WIN32

  // CHECK: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::win32_nt_handle);
  // CHECK-NEXT: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::win32_nt_dx12_resource);
  // CHECK-NEXT: memHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd);
  // CHECK-NEXT: memHandleDesc.set_handle_type(memHandleType);
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  memHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memHandleDesc.type = memHandleType;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryHandleTypeOpaqueWin32Kmt is not supported.
  // CHECK-NEXT: */
  memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
#endif // !NO_BUILD_TEST

  // CHECK: semHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_semaphore_handle_type::win32_nt_handle);
  // CHECK-NEXT: semHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_semaphore_handle_type::win32_nt_dx12_fence);
  // CHECK-NEXT: semHandleDesc.set_handle_type(sycl::ext::oneapi::experimental::external_semaphore_handle_type::opaque_fd);
  // CHECK-NEXT: semHandleDesc.set_handle_type(semHandleType);
  semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  semHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
  semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  semHandleDesc.type = semHandleType;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt is not supported.
  // CHECK-NEXT: */
  semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
#endif // !NO_BUILD_TEST

  // CHECK: memHandleDesc.set_flags(memHandleFlags);
  // CHECK-NEXT: memHandleDesc.set_flags(0);
  // CHECK-NEXT: memHandleDesc.set_flags(0);
  memHandleDesc.flags = memHandleFlags;
  memHandleDesc.flags = 0;
  memHandleDesc.flags = cudaExternalMemoryDedicated;

  // CHECK: semHandleDesc.set_flags(semHandleFlags);
  // CHECK-NEXT: semHandleDesc.set_flags(0);
  semHandleDesc.flags = semHandleFlags;
  semHandleDesc.flags = 0;

  // CHECK: memHandleDesc.set_res_size(9);
  // CHECK-NEXT: memHandleDesc.set_res_size(sizeInBytes);
  memHandleDesc.size = 9;
  memHandleDesc.size = sizeInBytes;

  // CHECK: mipmappedArrDesc.set_size(extent);
  mipmappedArrDesc.extent = extent;

  // CHECK: mipmappedArrDesc.set_image_type(mipFlags);
  // CHECK-NEXT: mipmappedArrDesc.set_image_type(sycl::ext::oneapi::experimental::image_type::mipmap);
  mipmappedArrDesc.flags = mipFlags;
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

  // CHECK: bufferDesc.set_flags(buffFlags);
  // CHECK-NEXT: bufferDesc.set_flags(0);
  bufferDesc.flags = buffFlags;
  bufferDesc.flags = 0;

  // CHECK: bufferDesc.set_mem_offset(offset);
  // CHECK-NEXT: bufferDesc.set_mem_offset(9);
  bufferDesc.offset = offset;
  bufferDesc.offset = 9;

  // CHECK: bufferDesc.set_res_size(sizeInBytes);
  // CHECK-NEXT: bufferDesc.set_res_size(9);
  bufferDesc.size = sizeInBytes;
  bufferDesc.size = 9;

  // CHECK: signalParams.set_value(value);
  // CHECK-NEXT: signalParams.set_value(0);
  signalParams.params.fence.value = value;
  signalParams.params.fence.value = 0;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  signalParams.params.nvSciSync.fence = fence;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  signalParams.params.nvSciSync.reserved = nvSciSync_reserved;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  signalParams.params.keyedMutex.key = key;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::flags is not supported.
  // CHECK-NEXT: */
  signalParams.flags = semSignalParamFlags;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::flags is not supported.
  // CHECK-NEXT: */
  signalParams.flags = cudaExternalSemaphoreSignalSkipNvSciBufMemSync;
#endif // !NO_BUILD_TEST

  // CHECK: waitParams.set_value(value);
  // CHECK-NEXT: waitParams.set_value(0);
  waitParams.params.fence.value = value;
  waitParams.params.fence.value = 0;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  waitParams.params.nvSciSync.fence = fence;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  waitParams.params.nvSciSync.reserved = nvSciSync_reserved;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  waitParams.params.keyedMutex.key = key;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  waitParams.params.keyedMutex.timeoutMs = timeoutMs;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::flags is not supported.
  // CHECK-NEXT: */
  waitParams.flags = semSignalParamFlags;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::flags is not supported.
  // CHECK-NEXT: */
  waitParams.flags = cudaExternalSemaphoreSignalSkipNvSciBufMemSync;
#endif // !NO_BUILD_TEST


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
  // CHECK: memHandleType = memHandleDesc.get_handle_type();
  // CHECK-NEXT: memHandleFlags = memHandleDesc.get_flags();
  // CHECK-NEXT: sizeInBytes = memHandleDesc.get_res_size();
  memHandleType = memHandleDesc.type;
  memHandleFlags = memHandleDesc.flags;
  sizeInBytes = memHandleDesc.size;

#ifdef _WIN32
  // CHECK-WINDOWS: win_nt_handle = semHandleDesc.get_win32_handle();
  // CHECK-WINDOWS-NEXT: obj = semHandleDesc.get_win32_obj_name();
  win_nt_handle = semHandleDesc.handle.win32.handle;
  obj = semHandleDesc.handle.win32.name;
#else
  // CHECK-LINUX: fd = semHandleDesc.get_fd_handle();
  fd = semHandleDesc.handle.fd;
#endif // _WIN32
  // CHECK: semHandleType = semHandleDesc.get_handle_type();
  // CHECK-NEXT: semHandleFlags = semHandleDesc.get_flags();
  semHandleType = semHandleDesc.type;
  semHandleFlags = semHandleDesc.flags;

  // CHECK: extent = mipmappedArrDesc.get_size();
  // CHECK-NEXT: mipFlags = mipmappedArrDesc.get_image_type();
  // CHECK-NEXT: formatDesc = mipmappedArrDesc.get_image_channel();
  // CHECK-NEXT: numLevels = mipmappedArrDesc.get_num_levels();
  extent = mipmappedArrDesc.extent;
  mipFlags = mipmappedArrDesc.flags;
  formatDesc = mipmappedArrDesc.formatDesc;
  numLevels = mipmappedArrDesc.numLevels;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalMemoryMipmappedArrayDesc::offset is not supported.
  // CHECK-NEXT: */
  offset = mipmappedArrDesc.offset;
#endif // !NO_BUILD_TEST

  // CHECK: buffFlags = bufferDesc.get_flags();
  // CHECK-NEXT: offset = bufferDesc.get_mem_offset();
  // CHECK-NEXT: sizeInBytes = bufferDesc.get_res_size();
  buffFlags = bufferDesc.flags;
  offset = bufferDesc.offset;
  sizeInBytes = bufferDesc.size;

  // CHECK: value = signalParams.get_value();
  value = signalParams.params.fence.value;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  fence = signalParams.params.nvSciSync.fence;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  nvSciSync_reserved = signalParams.params.nvSciSync.reserved;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  key = signalParams.params.keyedMutex.key;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreSignalParams::flags is not supported.
  // CHECK-NEXT: */
  semSignalParamFlags = signalParams.flags;
#endif // !NO_BUILD_TEST

  // CHECK: value = waitParams.get_value();
  value = waitParams.params.fence.value;
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  fence = waitParams.params.nvSciSync.fence;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::nvSciSync is not supported.
  // CHECK-NEXT: */
  nvSciSync_reserved = waitParams.params.nvSciSync.reserved;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  key = waitParams.params.keyedMutex.key;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::params::keyedMutex is not supported.
  // CHECK-NEXT: */
  timeoutMs = waitParams.params.keyedMutex.timeoutMs;
  // CHECK: /*
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaExternalSemaphoreWaitParams::flags is not supported.
  // CHECK-NEXT: */
  semWaitParamFlags = waitParams.flags;
#endif // !NO_BUILD_TEST


  /// calls
  // CHECK-WINDOWS: /*
  // CHECK-WINDOWS-NEXT: DPCT1136:{{[0-9]+}}: SYCL Bindless Images extension only supports importing external resource using NT handle on Windows. If assert(memHandleDesc.get_win32_handle()) fails, you may need to adjust the code to use (memHandleDesc.get_win32_handle()).
  // CHECK-WINDOWS-NEXT: */
  // CHECK: dpct::experimental::import_external_memory(&extMem, &memHandleDesc);
  // CHECK-NEXT: mipmap = new dpct::experimental::image_mem_wrapper(extMem, &mipmappedArrDesc);
  // CHECK-NEXT: devPtr = sycl::ext::oneapi::experimental::map_external_linear_memory(extMem, (&bufferDesc)->get_res_size(), (&bufferDesc)->get_mem_offset(), q_ct1);
  // CHECK-NEXT: sycl::ext::oneapi::experimental::release_external_memory(extMem, q_ct1);
  cudaImportExternalMemory(&extMem, &memHandleDesc);
  cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &mipmappedArrDesc);
  cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
  cudaDestroyExternalMemory(extMem);

  // CHECK-WINDOWS: /*
  // CHECK-WINDOWS-NEXT: DPCT1136:{{[0-9]+}}: SYCL Bindless Images extension only supports importing external resource using NT handle on Windows. If assert(semHandleDesc.get_win32_handle()) fails, you may need to adjust the code to use (semHandleDesc.get_win32_handle()).
  // CHECK-WINDOWS-NEXT: */
  // CHECK: dpct::experimental::import_external_semaphore(&extSem, &semHandleDesc);
  // CHECK-NEXT: dpct::experimental::signal_external_semaphore(&extSem, &signalParams, 1);
  // CHECK-NEXT: dpct::experimental::signal_external_semaphore(extSemArr, signalParamsArr, numExtSems, stream);
  // CHECK-NEXT: dpct::experimental::wait_external_semaphore(&extSem, &waitParams, 1);
  // CHECK-NEXT: dpct::experimental::wait_external_semaphore(extSemArr, waitParamsArr, numExtSems, stream);
  // CHECK-NEXT: delete extSem;
  cudaImportExternalSemaphore(&extSem, &semHandleDesc);
  cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1);
  cudaSignalExternalSemaphoresAsync(extSemArr, signalParamsArr, numExtSems, stream);
  cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1);
  cudaWaitExternalSemaphoresAsync(extSemArr, waitParamsArr, numExtSems, stream);
  cudaDestroyExternalSemaphore(extSem);

  return 0;
}
