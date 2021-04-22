cmake_minimum_required(VERSION 3.13) 
project(DISC CXX)
set(CMAKE_CXX_STANDARD 17)

option(WITH_UNITTESTS       "build unittests"      OFF)
option(WITH_PYTHON_BINDINGS "build python bindings" ON)
option(WITH_MPFR            "use MPFR backend"     OFF)

##############################################################################
# Library
##############################################################################
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

##############################################################################
# Dependencies
##############################################################################
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED) 

target_link_libraries(${PROJECT_NAME} INTERFACE 
        Boost::boost
        OpenMP::OpenMP_CXX)

if(WITH_MPFR) 

    find_package(MPFR REQUIRED)

    target_link_libraries(${PROJECT_NAME} INTERFACE Boost::boost ${MPFR_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} INTERFACE "WITH_MPFR=1")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    target_link_libraries(${PROJECT_NAME} INTERFACE quadmath Boost::boost)
    target_compile_definitions(${PROJECT_NAME} INTERFACE "WITH_QUADMATH=1")
    target_compile_options(${PROJECT_NAME} INTERFACE "-fext-numeric-literals")

    find_package(TBB)
    if (TBB_FOUND AND TBB_VERSION STREQUAL "2020.2") 
        target_compile_definitions(${PROJECT_NAME} INTERFACE "HAS_EXECUTION_POLICIES=1")
        target_link_libraries(${PROJECT_NAME} INTERFACE TBB::tbb)
    endif()
endif()

##############################################################################
# bindings
##############################################################################
if (WITH_PYTHON_BINDINGS)
    add_subdirectory(src/bindings/python)
endif()

##############################################################################
# unittests
##############################################################################
if (WITH_UNITTESTS)
    add_subdirectory(unittests)
endif()

##############################################################################
# installation
##############################################################################

install(TARGETS               ${PROJECT_NAME}
        EXPORT                ${PROJECT_NAME}_exports
        INCLUDES  DESTINATION include/${PROJECT_NAME}
)
install(EXPORT      ${PROJECT_NAME}_exports
        FILE        ${PROJECT_NAME}-config.cmake 
        NAMESPACE   ${PROJECT_NAME}::
        DESTINATION share/cmake
)
