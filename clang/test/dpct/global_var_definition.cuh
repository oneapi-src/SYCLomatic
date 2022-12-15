#ifndef GLOBAL_VAR_DEFINITION_CUH
#define GLOBAL_VAR_DEFINITION_CUH

//CHECK:inline dpct::constant_memory<float, 1> c_clusters(34);
__constant__ float c_clusters[34];

#endif