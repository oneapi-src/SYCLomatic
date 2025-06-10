.. _migration_examples:

Migration Examples
==================

External Memory Interoperability
--------------------------------

|tool_name| now supports experimental migration of DirectX (11/12) and
Vulkan Interoperability with CUDA using option
``--use-experimental-features=bindless_images``.

Supported CUDA APIs
*******************
* **cudaGraphicsMapResources**
* **cudaGraphicsResourceGetMappedPointer**
* **cudaGraphicsSubResourceGetMappedArray**
* **cudaGraphicsResourceGetMappedMipmappedArray**
* **cudaGraphicsUnmapResources**
* **cudaGraphicsUnregisterResource**
* **cudaGraphicsResourceSetMapFlags**
* **cudaGraphicsD3D11RegisterResource**
* **cudaDestroyExternalMemory**
* **CudaExternalMemoryGetMappedBuffer**
* **cudaExternalMemoryGetMappedMipmappedArray**
* **cudaImportExternalMemory**
* **cudaDestroyExternalSemaphore**
* **cudaImportExternalSemaphore**
* **cudaSignalExternalSemaphoresAsync**
* **cudaWaitExternalSemaphoresAsync**

Examples
********

DirectX-CUDA Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA

.. code-block:: none

   // Create a resource using DirectX APIs
   ID3D11Resource *d3dResource

   cudaGraphicsResource_t cudaResource;

   // Register the DirectX resource with CUDA
   cudaGraphicsD3D11RegisterResource(&cudaResource, d3dResource, 
                                     cudaGraphicsRegisterFlagsNone);

   // Set the flags for CUDA resource mapping
   cudaGraphicsResourceSetMapFlags(cudaResource, cudaGraphicsMapFlagsNone);

   // Map the CUDA resource for access
   cudaGraphicsMapResources(1, &cudaResource);

   // Get the mapped array from the CUDA resource
   cudaArray_t cudaArr;
   cudaGraphicsSubResourceGetMappedArray(&cudaArr, cudaResource, 0, 0);

   // Unmap the CUDA resource
   cudaGraphicsUnmapResources(1, &cudaResource);

   // Unregister the CUDA resource
   cudaGraphicsUnregisterResource(cudaResource);

Migrated Code

.. code-block:: none

   // Create a resource using DirectX APIs
   ID3D11Resource *d3dResource

   dpct::experimental::external_mem_wrapper_ptr cudaResource;

   // Register the DirectX resource with CUDA
   cudaResource = new dpct::experimental::external_mem_wrapper(d3dResource,
                                                               0);

   // Set the flags for CUDA resource mapping
   /*
   DPCT1026:1: The call to cudaGraphicsResourceSetMapFlags was removed because this functionality is deprecated in DX12 and hence is not supported in SYCL.
   */

   // Map the CUDA resource for access
   dpct::experimental::map_resources(1, &cudaResource);

   // Get the mapped array from the CUDA resource
   dpct::experimental::image_mem_wrapper_ptr cudaArr;
   cudaArr = cudaResource->get_sub_resource_mapped_array(0, 0);

   // Unmap the CUDA resource
   dpct::experimental::unmap_resources(1, &cudaResource);

   // Unregister the CUDA resource
   delete cudaResource;

Vulkan-CUDA Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA

.. code-block:: none

   VkDevice device = {};

   // Create a texture resource using Vulkan APIs
   VkImage vkTexture;
   VkImageCreateInfo imageCreateInfo = {}; // init and fill texture descriptor
   vkCreateImage(device, &imageCreateInfo, nullptr, &vkTexture);

   // Create a semaphore resource using Vulkan APIs
   VkSemaphore vkSemaphore;
   VkSemaphoreCreateInfo semaphoreInfo = {}; // init and fill semaphore descriptor
   vkCreateSemaphore(device, &semaphoreInfo, nullptr, &vkSemaphore);

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   cudaExternalMemoryHandleDesc memHandleDesc;
   cudaExternalMemoryMipmappedArrayDesc mipmapDesc;
   cudaExternalMemory_t externalMemory;
   cudaExternalSemaphoreHandleDesc semHandleDesc;
   cudaExternalSemaphore_t externalSemaphore;
   cudaExternalSemaphoreWaitParams waitParams;
   cudaExternalSemaphoreSignalParams signalParams;


   // Import the memory from external resource (vkTexture)
   cudaImportExternalMemory(&externalMemory, &memHandleDesc);
   // Import the semaphore from external resource (vkSemaphore)
   cudaImportExternalSemaphore(&externalSemaphore, &semHandleDesc);

   // Wait on the semaphore using external resource
   cudaWaitExternalSemaphoresAsync(&externalSemaphore, &waitParams, 1, stream);

   // Get the mapped array from the CUDA resource
   cudaMipmappedArray_t cudaMipmappedArray = nullptr;
   cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedArray, externalMemory, &mipmapDesc);

   // Signal the semaphore using external resource
   cudaSignalExternalSemaphoresAsync(&externalSemaphore, &signalParams, 1, stream);

   // Retrieve the tex data as a cudaArray from cudaMipmappedArray
   cudaArray_t cudaArr;
   cudaGetMipmappedArrayLevel(&cudaArr, cudaMipmappedArray, 0);

   // Destroy the CUDA resources
   cudaDestroyExternalMemory(externalMemory);
   cudaDestroyExternalSemaphore(externalSemaphore);

Migrated Code

.. code-block:: none

   VkDevice device = {};

   // Create a texture resource using Vulkan APIs
   VkImage vkTexture;
   VkImageCreateInfo imageCreateInfo = {}; // init and fill texture descriptor
   vkCreateImage(device, &imageCreateInfo, nullptr, &vkTexture);

   // Create a semaphore resource using Vulkan APIs
   VkSemaphore vkSemaphore;
   VkSemaphoreCreateInfo semaphoreInfo = {}; // init and fill semaphore descriptor
   vkCreateSemaphore(device, &semaphoreInfo, nullptr, &vkSemaphore);

   dpct::queue_ptr stream;
   stream = dpct::get_current_device().create_queue();

   dpct::experimental::external_mem_handle_desc memHandleDesc;
   dpct::experimental::external_mem_img_desc mipmapDesc;
   sycl::ext::oneapi::experimental::external_mem externalMemory;
   dpct::experimental::external_sem_handle_desc semHandleDesc;
   dpct::experimental::external_sem_wrapper_ptr externalSemaphore;
   dpct::experimental::external_sem_params waitParams;
   dpct::experimental::external_sem_params signalParams;


   // Import the memory from external resource (vkTexture)
   dpct::experimental::import_external_memory(&externalMemory, &memHandleDesc));
   // Import the semaphore from external resource (vkSemaphore)
   dpct::experimental::import_external_semaphore(&externalSemaphore, &semHandleDesc);

   // Wait on the semaphore using external resource
   dpct::experimental::wait_external_semaphore(&externalSemaphore, &waitParams, 1, stream);

   // Get the mapped array from the CUDA resource
   dpct::experimental::image_mem_wrapper_ptr cudaMipmappedArray = nullptr;
   cudaMipmappedArray = new dpct::experimental::image_mem_wrapper(externalMemory, &mipmapDesc);

   // Signal the semaphore using external resource
   dpct::experimental::signal_external_semaphore(&externalSemaphore, &signalParams, 1, stream);

   // Retrieve the tex data as a cudaArray from cudaMipmappedArray
   dpct::experimental::image_mem_wrapper_ptr cudaArr;
   cudaArr = cudaMipmappedArray->get_mip_level(0);

   // Destroy the CUDA resources
   sycl::ext::oneapi::experimental::release_external_memory(
           externalMemory, dpct::get_in_order_queue());
   delete externalSemaphore;


Additional Migration Examples
-----------------------------

Example: Migrate QuickSilver to SYCL Version
********************************************

View a `list of detailed steps <https://github.com/oneapi-src/SYCLomatic-test/tree/SYCLomatic/third-party-programs/Velocity-Bench/QuickSilver>`__ to migrate CUDA version of
QuickSilver to SYCL version.

Example: Migrate cudaSift to SYCL Version
*****************************************

View a `list of detailed steps <https://github.com/oneapi-src/SYCLomatic-test/tree/SYCLomatic/third-party-programs/Velocity-Bench/cudaSift>`__ to migrate CUDA version
of cudaSift to SYCL version.

Example: Migrate hplinpack to SYCL Version
******************************************

View a `list of detailed steps <https://github.com/oneapi-src/SYCLomatic-test/tree/SYCLomatic/third-party-programs/Velocity-Bench/hplinpack>`__ to migrate CUDA version
of hplinpack to SYCL version.

Example: Migrate bitcracker to SYCL Version
*******************************************

View a `list of detailed steps <https://github.com/oneapi-src/SYCLomatic-test/tree/SYCLomatic/third-party-programs/Velocity-Bench/bitcracker>`__ to migrate CUDA version
of bitcracker to SYCL version.

