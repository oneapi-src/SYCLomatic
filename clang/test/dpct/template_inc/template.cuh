template <typename T>
__global__ void foo(T t){
    int a = blockIdx.x;
}

template <typename T>
class B{};

template< template<typename, int> class T, int I, typename Q>
void instantiate(Q q){int x = I;}