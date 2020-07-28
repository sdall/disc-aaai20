cmake_minimum_required(VERSION 3.9) 
project(disc CXX)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_UNITTESTS OFF "build unittests")
option(WITH_MPFR_BACKEND OFF "use MPFR backend")

##############################################################################
# Library
##############################################################################
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)
target_include_directories(${PROJECT_NAME} INTERFACE include)

##############################################################################
# Dependencies
##############################################################################
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED) 
find_package(TBB REQUIRED)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE 
        OpenMP::OpenMP_CXX
        ${TBB_LIBRARIES}
        ${Boost_INCLUDES}
        atomic)
        
target_compile_definitions(${PROJECT_NAME} 
    INTERFACE 
        WITH_EXECUTION_POLICIES=1)

if(WITH_MPFR_BACKEND) 
    find_package(MPFR REQUIRED)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${MPFR_LIBRARIES})
else()
    find_library(QUADMATH_LIBRARY NAMES quadmath libquadmath)
    target_link_libraries(${PROJECT_NAME} INTERFACE quadmath)
endif()

##############################################################################
# unittests
##############################################################################

if (BUILD_UNITTESTS)
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