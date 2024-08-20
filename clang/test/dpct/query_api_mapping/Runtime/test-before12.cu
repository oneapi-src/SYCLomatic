// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6

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
