
ENTRY_RENAMED("cudaCreateChannelDesc", "dpct::create_image_channel")
ENTRY_RENAMED("cudaCreateChannelDescHalf", "dpct::create_image_channel<cl::sycl::cl_half>")
ENTRY_RENAMED("cudaUnbindTexture", "dpct::dpct_detach_image")
ENTRY_RENAMED("cudaFreeArray", "dpct::dpct_free")
ENTRY_RENAMED("cudaDestroyTextureObject", "dpct::dpct_free")
ENTRY_RENAMED("cudaGetTextureObjectResourceDesc", "dpct::dpct_get_image_data")
ENTRY_RENAMED("cudaGetTextureObjectTextureDesc", "dpct::dpct_get_image_info")
ENTRY_TEXTURE("tex1D", "dpct::dpct_read_image")
ENTRY_TEXTURE("tex2D", "dpct::dpct_read_image")
ENTRY_TEXTURE("tex3D", "dpct::dpct_read_image")
ENTRY_TEXTURE("tex1Dfetch", "dpct::dpct_read_image")
ENTRY_REORDER("cudaBindTexture", "dpct::dpct_attach_image",
              1 /*ref to tex object*/, 2 /*source pointer*/,
              3 /*ref to channel desc*/, 4 /*size in bytes*/)
ENTRY_REORDER("cudaMallocArray", "dpct::dpct_malloc_matrix",
              0 /*pointer to array*/, 1 /*pointer to channel desc*/,
              2 /*width*/, 3 /*height*/)
ENTRY_REORDER("cudaMemcpyToArray", "dpct::dpct_memcpy_to_matrix",
              0 /*ref of array*/, 1 /*x offset*/, 2 /*y offset*/,
              3 /*src pointer*/, 4 /*bytes to copy*/)
ENTRY_REORDER("cudaBindTextureToArray", "dpct::dpct_attach_image",
              0 /*ref of texture object*/, 1 /*ref of array*/)
ENTRY_REORDER("cudaCreateTextureObject", "dpct::dpct_create_image",
              0 /*ref of texture object*/, 1 /*ref of data desc*/,
              2 /*ref of info desc*/)
ENTRY_UNSUPPORTED("cudaGetTextureObjectResourceViewDesc",
                  Diagnostics::NOTSUPPORTED)