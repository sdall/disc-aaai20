cmake_minimum_required(VERSION 3.8) 
project(disc CXX)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_UNITTESTS OFF "build unittests")

##############################################################################
# Dependencies
##############################################################################
find_package(OpenMP REQUIRED)
find_library(QUADMATH_LIBRARY NAMES quadmath)

##############################################################################
# Library
##############################################################################
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/>
)

target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX atomic)

if (QUADMATH_LIBRARY)
    target_link_libraries(${PROJECT_NAME} INTERFACE quadmath)
    target_compile_definitions(${PROJECT_NAME} INTERFACE WITH_QUADMATH=1)
    # target_compile_options(${PROJECT_NAME} INTERFACE "-fext-numeric-literals")
endif()

install(DIRECTORY include/     DESTINATION include)    
install(DIRECTORY thirdparty/  DESTINATION include)    
install(TARGETS               ${PROJECT_NAME}
        EXPORT                ${PROJECT_NAME}_exports
        LIBRARY   DESTINATION lib
        ARCHIVE   DESTINATION lib
        RUNTIME   DESTINATION bin
        INCLUDES  DESTINATION include
        # DIRECTORY DESTINATION include
)
install(EXPORT      ${PROJECT_NAME}_exports
        FILE        ${PROJECT_NAME}-config.cmake
        NAMESPACE   sd::
        DESTINATION share/cmake
)

##############################################################################
# Targets
##############################################################################
if (BUILD_UNITTESTS)
    add_subdirectory(unittests)
endif()
