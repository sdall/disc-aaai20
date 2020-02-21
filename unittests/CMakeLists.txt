add_executable(test-bitcontainer bitcontainer/test-bitset.cxx)
target_link_libraries(test-bitcontainer PUBLIC disc)
target_include_directories(test-bitcontainer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
