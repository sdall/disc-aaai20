cmake_minimum_required(VERSION 3.9) 
project(disc CXX)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_UNITTESTS OFF "build unittests")

##############################################################################
# Dependencies
##############################################################################
find_package(OpenMP REQUIRED)
find_library(QUADMATH_LIBRARY NAMES quadmath)
find_package(Boost 1.5.0 COMPONENTS math) 

##############################################################################
# Library
##############################################################################
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)

target_include_directories(${PROJECT_NAME} INTERFACE ${Boost_INCLUDES})

if (OpenMP_FOUND)
    find_library(atomic REQUIRED)
    target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX atomic)
endif()

if (QUADMATH_LIBRARY)
    target_link_libraries(${PROJECT_NAME} INTERFACE quadmath)
    target_compile_definitions(${PROJECT_NAME} INTERFACE WITH_QUADMATH=1)
endif()

if (BUILD_UNITTESTS)
    add_subdirectory(unittests)
endif()

##############################################################################
# installation
##############################################################################

install(TARGETS               ${PROJECT_NAME}
        EXPORT                ${PROJECT_NAME}_exports
        INCLUDES  DESTINATION include
)
install(EXPORT      ${PROJECT_NAME}_exports
        FILE        ${PROJECT_NAME}-config.cmake
        NAMESPACE   ${PROJECT_NAME}::
        DESTINATION share/cmake
)

