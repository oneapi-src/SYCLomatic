// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8, cuda-12.9
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8, v12.9

/// Texture Reference Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaBindTexture | FileCheck %s -check-prefix=CUDABINDTEXTURE
// CUDABINDTEXTURE: CUDA API:
// CUDABINDTEXTURE-NEXT:   cudaBindTexture(ps /*size_t **/, ptr /*const textureReference **/,
// CUDABINDTEXTURE-NEXT:                   pv /*const void **/, pc /*const cudaChannelFormatDesc **/,
// CUDABINDTEXTURE-NEXT:                   s /*size_t*/);
// CUDABINDTEXTURE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDABINDTEXTURE-NEXT:   ptr->attach(pv, s, *pc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaBindTexture2D | FileCheck %s -check-prefix=CUDABINDTEXTURE2D
// CUDABINDTEXTURE2D: CUDA API:
// CUDABINDTEXTURE2D-NEXT:   cudaBindTexture2D(ps /*size_t **/, ptr /*const textureReference **/,
// CUDABINDTEXTURE2D-NEXT:                   pv /*const void **/, pc /*const cudaChannelFormatDesc **/,
// CUDABINDTEXTURE2D-NEXT:                   s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// CUDABINDTEXTURE2D-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDABINDTEXTURE2D-NEXT:   ptr->attach(pv, s1, s2, s3, *pc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaBindTextureToArray | FileCheck %s -check-prefix=CUDABINDTEXTURETOARRAY
// CUDABINDTEXTURETOARRAY: CUDA API:
// CUDABINDTEXTURETOARRAY-NEXT:   cudaBindTextureToArray(ptr /*const textureReference **/,
// CUDABINDTEXTURETOARRAY-NEXT:                          a /*const cudaArray_t*/,
// CUDABINDTEXTURETOARRAY-NEXT:                          pc /*const cudaChannelFormatDesc **/);
// CUDABINDTEXTURETOARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDABINDTEXTURETOARRAY-NEXT:   ptr->attach(a, *pc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaUnbindTexture | FileCheck %s -check-prefix=CUDAUNBINDTEXTURE
// CUDAUNBINDTEXTURE: CUDA API:
// CUDAUNBINDTEXTURE-NEXT:   cudaUnbindTexture(ptr /*const textureReference **/);
// CUDAUNBINDTEXTURE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDAUNBINDTEXTURE-NEXT:   ptr->detach();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaBindTextureToMipmappedArray | FileCheck %s -check-prefix=cudaBindTextureToMipmappedArray
// cudaBindTextureToMipmappedArray: CUDA API:
// cudaBindTextureToMipmappedArray-NEXT:   static texture<float4, 3> tex3;
// cudaBindTextureToMipmappedArray-NEXT:   cudaMipmappedArray_t pMipMapArr;
// cudaBindTextureToMipmappedArray-NEXT:   cudaBindTextureToMipmappedArray(tex3 /*const struct texture<T, dim, readMode>*/, pMipMapArr /*cudaMipmappedArray_const_t*/);
// cudaBindTextureToMipmappedArray-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaBindTextureToMipmappedArray-NEXT:   dpct::experimental::bindless_image_wrapper<sycl::float4, 3> tex3;
// cudaBindTextureToMipmappedArray-NEXT:   dpct::experimental::image_mem_wrapper_ptr pMipMapArr;
// cudaBindTextureToMipmappedArray-NEXT:   tex3.attach(pMipMapArr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateChannelDescHalf | FileCheck %s -check-prefix=cudaCreateChannelDescHalf
// cudaCreateChannelDescHalf: CUDA API:
// cudaCreateChannelDescHalf-NEXT:   cudaChannelFormatDesc halfChn = cudaCreateChannelDescHalf();
// cudaCreateChannelDescHalf-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaCreateChannelDescHalf-NEXT:   dpct::image_channel halfChn = dpct::image_channel::create<sycl::half>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateChannelDescHalf1 | FileCheck %s -check-prefix=cudaCreateChannelDescHalf1
// cudaCreateChannelDescHalf1: CUDA API:
// cudaCreateChannelDescHalf1-NEXT:   cudaChannelFormatDesc half1Chn = cudaCreateChannelDescHalf1();
// cudaCreateChannelDescHalf1-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaCreateChannelDescHalf1-NEXT:   dpct::image_channel half1Chn = dpct::image_channel::create<sycl::half>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateChannelDescHalf2 | FileCheck %s -check-prefix=cudaCreateChannelDescHalf2
// cudaCreateChannelDescHalf2: CUDA API:
// cudaCreateChannelDescHalf2-NEXT:   cudaChannelFormatDesc half2Chn = cudaCreateChannelDescHalf2();
// cudaCreateChannelDescHalf2-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaCreateChannelDescHalf2-NEXT:   dpct::image_channel half2Chn = dpct::image_channel::create<sycl::half2>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateChannelDescHalf4 | FileCheck %s -check-prefix=cudaCreateChannelDescHalf4
// cudaCreateChannelDescHalf4: CUDA API:
// cudaCreateChannelDescHalf4-NEXT:   cudaChannelFormatDesc half4Chn = cudaCreateChannelDescHalf4();
// cudaCreateChannelDescHalf4-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaCreateChannelDescHalf4-NEXT:   dpct::image_channel half4Chn = dpct::image_channel::create<sycl::half4>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateSurfaceObject | FileCheck %s -check-prefix=cudaCreateSurfaceObject
// cudaCreateSurfaceObject: CUDA API:
// cudaCreateSurfaceObject-NEXT:   cudaCreateSurfaceObject(&surf /*cudaSurfaceObject_t* */, &resDesc /*const cudaResourceDesc* */);
// cudaCreateSurfaceObject-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaCreateSurfaceObject-NEXT:   surf = dpct::experimental::create_bindless_image(resDesc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDestroySurfaceObject | FileCheck %s -check-prefix=cudaDestroySurfaceObject
// cudaDestroySurfaceObject:  CUDA API:
// cudaDestroySurfaceObject-NEXT:    cudaDestroySurfaceObject(surf /*cudaSurfaceObject_t*/);
// cudaDestroySurfaceObject-NEXT:  Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaDestroySurfaceObject-NEXT:    dpct::experimental::destroy_bindless_image(surf, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetSurfaceObjectResourceDesc | FileCheck %s -check-prefix=cudaGetSurfaceObjectResourceDesc
// cudaGetSurfaceObjectResourceDesc: CUDA API:
// cudaGetSurfaceObjectResourceDesc-NEXT:   cudaGetSurfaceObjectResourceDesc(&resDesc /*cudaResourceDesc* */, surf /*cudaSurfaceObject_t*/);
// cudaGetSurfaceObjectResourceDesc-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// cudaGetSurfaceObjectResourceDesc-NEXT:   resDesc = dpct::experimental::get_data(surf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphAddDependencies | FileCheck %s -check-prefix=cudaGraphAddDependencies
// cudaGraphAddDependencies: CUDA API:
// cudaGraphAddDependencies-NEXT:   cudaGraphAddDependencies(graph /*cudaGraph_t*/, node4 /*const cudaGraphNode_t* */, node5 /*const cudaGraphNode_t* */, 10 /*size_t*/);
// cudaGraphAddDependencies-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphAddDependencies-NEXT:   dpct::experimental::add_dependencies(graph, node4, node5, 10);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphAddEmptyNode | FileCheck %s -check-prefix=cudaGraphAddEmptyNode
// cudaGraphAddEmptyNode: CUDA API:
// cudaGraphAddEmptyNode-NEXT:   cudaGraphAddEmptyNode(&node /*cudaGraphNode_t* */, graph /*cudaGraph_t*/, node4 /*const cudaGraphNode_t* */, 10 /*size_t*/);
// cudaGraphAddEmptyNode-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphAddEmptyNode-NEXT:   dpct::experimental::add_empty_node(&node, graph, node4, 10);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphDestroy | FileCheck %s -check-prefix=cudaGraphDestroy
// cudaGraphDestroy: CUDA API:
// cudaGraphDestroy-NEXT:   cudaGraphDestroy(graph /*cudaGraph_t*/);
// cudaGraphDestroy-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphDestroy-NEXT:   delete (graph);
