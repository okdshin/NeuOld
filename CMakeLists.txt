cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(neu)
enable_testing()

include(CheckIncludeFiles)
include (FindPkgConfig)
include(CheckCXXSourceCompiles)

# build type
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -Og -g")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -Og -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wno-unused-parameter -Wno-deprecated-declarations -std=c++14 -O2")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -Wno-unused-parameter -Werror -O2")

find_package( Boost COMPONENTS system timer filesystem REQUIRED)
if( NOT Boost_TIMER_FOUND )
	message( SEND_ERROR "Required package Boost.Timer is not detected." )
endif (NOT Boost_TIMER_FOUND)
if( NOT Boost_FILESYSTEM_FOUND )
	message( SEND_ERROR "Required package Boost.Filesystem is not detected." )
endif (NOT Boost_FILESYSTEM_FOUND)

find_package( BoostCompute REQUIRED )
if( NOT BoostCompute_FOUND )
	message( SEND_ERROR "Required package Boost.Compute is not detected." )
endif (NOT BoostCompute_FOUND)

find_package( OpenCL REQUIRED )
if( NOT OpenCL_FOUND )
	message( SEND_ERROR "Required package OpenCL is not detected." )
endif (NOT OpenCL_FOUND)

find_package( freeimageplus REQUIRED )
if( NOT FREEIMAGEPLUS_FOUND )
	message( SEND_ERROR "Required package FreeImagePlus is not detected." )
endif (NOT FREEIMAGEPLUS_FOUND)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${BoostCompute_INCLUDE_DIRS} ${FREEIMAGEPLUS_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS} ${OpenCL_LIBRARY_DIRS} ${FREEIMAGEPLUS_LIBRARY_RELEASE})

install( DIRECTORY ./include/neu/ DESTINATION include/neu/ FILES_MATCHING PATTERN "*.hpp" )

subdirs( test example )
