cmake_minimum_required(VERSION 3.20)

project(loom_kernel_ledger LANGUAGES C)

find_package(loomc CONFIG REQUIRED)

add_executable(query_launch_info query_launch_info.c)
target_link_libraries(query_launch_info PRIVATE loomc::loomc)
