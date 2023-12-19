function (user_function file1 file2)
  message("Doing nothing in function1")
endfunction

user_function(file1.cu
              file2.cu
)

user_function(filea.cpp fileb.cu file.cpp)

User_Function(file1.cu file2.cu)

cmake_language(CALL user_function file1.cu, file2.cu)
