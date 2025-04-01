# baz.cpp is a C++ file
out = func("bar.cu", "baz.cpp")

# foo.cpp is a C++ file with CUDA syntax
out = func("foo.cpp", "bar.cu")

# src/bar.cpp is a C++ file with CUDA syntax
# dst/bar.cpp is a C++ file
out = func("src/bar.cpp", "dst/bar.cpp")

("foo.cpp", ("bar.cu", ("baz.cpp"), ("foo.cpp", ), ["foo.cpp"], ("foo.cpp"), {"foo.cpp"}, "foo.cpp"))
["foo.cpp", ["bar.cu", ["baz.cpp"], ["foo.cpp", ], ["foo.cpp"], ("foo.cpp"), {"foo.cpp"}, "foo.cpp"]]
{"foo.cpp", {"bar.cu", {"baz.cpp"}, {"foo.cpp", }, ["foo.cpp"], ("foo.cpp"), {"foo.cpp"}, "foo.cpp"}}
