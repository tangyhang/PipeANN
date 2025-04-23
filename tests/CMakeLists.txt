# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME} -ltcmalloc)

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME} aio -ltcmalloc)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)

add_executable(gen_tags gen_tags.cpp)
target_link_libraries(gen_tags ${PROJECT_NAME} aio)

add_executable(convert_bin convert_bin.cpp)
target_link_libraries(convert_bin ${PROJECT_NAME} aio)

add_executable(change_pts change_pts.cpp)
target_link_libraries(change_pts ${PROJECT_NAME} aio)

add_executable(reset_bin reset_bin.cpp)
target_link_libraries(reset_bin ${PROJECT_NAME} aio)

add_executable(tag_to_partition tag_to_partition.cpp)
target_link_libraries(tag_to_partition ${PROJECT_NAME} jemalloc aio)

add_executable(convert_spann_tag convert_spann_tag.cpp)
target_link_libraries(convert_spann_tag ${PROJECT_NAME} jemalloc aio)

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} jemalloc aio)

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME} jemalloc aio)

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} aio)
