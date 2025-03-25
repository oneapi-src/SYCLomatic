# baz.cpp is a C++ file
out = func("bar.dp.cpp", "baz.cpp")

# foo.cpp is a C++ file with CUDA syntax
out = func("foo.cpp.dp.cpp", "bar.dp.cpp")

# src/bar.cpp is a C++ file with CUDA syntax
# dst/bar.cpp is a C++ file
out = func("src/bar.cpp.dp.cpp", "dst/bar.cpp")

("foo.cpp.dp.cpp", ("bar.dp.cpp", ("baz.cpp"), ("foo.cpp.dp.cpp", ), ["foo.cpp.dp.cpp"], ("foo.cpp.dp.cpp"), {"foo.cpp.dp.cpp"}, "foo.cpp.dp.cpp"))
["foo.cpp.dp.cpp", ["bar.dp.cpp", ["baz.cpp"], ["foo.cpp.dp.cpp", ], ["foo.cpp.dp.cpp"], ("foo.cpp.dp.cpp"), {"foo.cpp.dp.cpp"}, "foo.cpp.dp.cpp"]]
{"foo.cpp.dp.cpp", {"bar.dp.cpp", {"baz.cpp"}, {"foo.cpp.dp.cpp", }, ["foo.cpp.dp.cpp"], ("foo.cpp.dp.cpp"), {"foo.cpp.dp.cpp"}, "foo.cpp.dp.cpp"}}
