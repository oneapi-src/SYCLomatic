CMake_minimum_required(VERSION 3.10)
Project(foo-bar LANGUAGES CXX CUDA)

Find_Package(CUDA REQUIRED)
Find_Package(CUDA)
