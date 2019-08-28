cmake_minimum_required(VERSION 2.8)
project(MonoVO)

# Set our build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -m64 -O0 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

# Include our cmake files
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules/)

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.7 REQUIRED)
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem thread)

message(${EIGEN3_INCLUDE_DIR})

# Include directory for headers
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

##################################################
# Make binary for the main library
##################################################
# Set sourcefiles
LIST(APPEND SOURCEFILES
  src/core/tracking.cpp
  src/core/reconstruction.cpp
  src/core/imageProc.cpp
  src/core/time.cpp
  src/core/utils.cpp
  src/main.cpp
)

# Set link libraries
LIST(APPEND LINK_LIBS
  ${EIGEN3_LIBS}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

##################################################
# Make binary for main method ROS
##################################################
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCEFILES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})