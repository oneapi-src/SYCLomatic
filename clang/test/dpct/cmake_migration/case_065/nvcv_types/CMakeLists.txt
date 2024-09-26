
# nvcv_types private implementation
add_subdirectory(priv)

add_library(nvcv_types SHARED
    Image.cpp
    ${CMAKE_SOURCE_DIR}/priv/foo.cpp
    ${CMAKE_SOURCE_DIR}/../infrastructure/Utilities.cpp
    ${CMAKE_SOURCE_DIR}/../../infrastructure/bar.cpp
)
