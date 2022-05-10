#ifndef __HEAD_MIGRATION_MERGE__H__
#define __HEAD_MIGRATION_MERGE__H__

#ifdef MACROA
//CHECK:void Hello_MacroA(){
__global__ void Hello_MacroA(){
}
#else
//CHECK:void Hello_MacroB(){
__global__ void Hello_MacroB(){
}
#endif

#endif
