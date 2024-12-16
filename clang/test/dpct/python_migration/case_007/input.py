# baz.cpp is a C++ file
out = func("bar.cu", "baz.cpp")
# foo.cpp is a C++ file with CUDA syntax
out = func("foo.cpp", "bar.cu")
out = func(["foo.cpp", "bar.cu"])
