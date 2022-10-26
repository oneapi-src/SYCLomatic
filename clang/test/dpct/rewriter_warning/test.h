#define cudaCheckError(msg) {                                                \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "%s: %i: %s: %s.\n",                                 \
                __FILE__, __LINE__, msg, cudaGetErrorString(err));           \
        exit(-1);                                                            \
    } }