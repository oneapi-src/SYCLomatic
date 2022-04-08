// CHECK: #ifndef __ATOMIC_NO_WARNING_H__
// CHECK-NEXT: #define __ATOMIC_NO_WARNING_H__
// CHECK-NEXT: #define ATOMIC_ADD( x, v )  c2s::atomic_fetch_add( &x, v );
// CHECK-NEXT: #endif // __ATOMIC_NO_WARNING_H__
#ifndef __ATOMIC_NO_WARNING_H__
#define __ATOMIC_NO_WARNING_H__
#define ATOMIC_ADD( x, v )  atomicAdd( &x, v );
#endif // __ATOMIC_NO_WARNING_H__