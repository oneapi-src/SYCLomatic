# this doesn't provide foo
# suggest to assign bar on your own
set(FOO_HEADERS_CUDA foo.h)
set(FOO_SOURCES_CUDA foo.cu)
target_compile_features(bar PUBLIC cxx_std_17) # don't bump
target_link_libraries(bar PRIVATE
    FOO
    ${bar_EXTRA_LIBS}
    )
message("###Here just a test msg.\n")
