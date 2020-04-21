// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include"
#include "../template_inc/template.cuh"

// This case is for preventing migration crash, no check here.
namespace NA {
template <typename T, int X> class A{
    B<T> b;
public:
    int fooInA(){
        int a = X;
        // Since the migration of foo is not correct because template.cuh is not in in-root,
        // we don't do check here.
        foo<<<1,2,3>>>(b);
        return 0;
    };
};
int foob(){
    int x;
    instantiate<A, 5>(x);
}
}

