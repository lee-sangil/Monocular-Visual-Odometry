cmake_minimum_required(VERSION 2.8)
project(MonoVO)

SET(DEBUG false)

# Set our build type
IF(${DEBUG})
  SET(CMAKE_BUILD_TYPE DEBUG)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg -O0")
ELSE()
  SET(CMAKE_BUILD_TYPE RELEASE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fomit-frame-pointer -flto -fopenmp")
ENDIF()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "Using flag -std=c++11.")

# Enable compile optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Include our cmake files
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules/)

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
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
# LIST(APPEND SOURCEFILES
#   src/core/MVO.cpp
#   src/core/tracking.cpp
#   src/core/mapping.cpp
#   src/core/imageProc.cpp
#   src/core/time.cpp
#   src/core/utils.cpp
#   src/visualization/plot.cpp
# )

# # Set link libraries
# LIST(APPEND LINK_LIBS
#   ${EIGEN3_LIBS}
#   ${OpenCV_LIBS}
#   ${Boost_LIBRARIES}
# )

##################################################
# Make binary for main method ROS
##################################################
add_library(${PROJECT_NAME} SHARED
  src/core/MVO.cpp
  src/core/tracking.cpp
  src/core/mapping.cpp
  src/core/function.cpp
  src/core/imageProc.cpp
  src/core/time.cpp
  src/core/random.cpp
  src/core/DepthFilter.cpp
  src/visualization/plot.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${EIGEN3_LIBS}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

add_executable(main src/main.cpp)
target_link_libraries(main
  ${PROJECT_NAME}
)
