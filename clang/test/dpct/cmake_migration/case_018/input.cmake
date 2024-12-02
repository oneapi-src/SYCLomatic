cuda_compile(obj1 src1.cu src2.cu)

cuda_compile(obj2 f1.cu f2.cu f3.cu STATIC)

cuda_compile(obj3 f1.cu f2.cu f3.cu SHARED)

cuda_compile(obj4 a.cu b.cu STATIC OPTIONS -O3 --profile)

cuda_compile(obj5 a.cu b.cu SHARED OPTIONS -O3 --profile)

cuda_compile(obj6 a.cu b.cu OPTIONS -O3 --profile)
