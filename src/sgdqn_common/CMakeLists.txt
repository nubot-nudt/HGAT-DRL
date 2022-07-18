cmake_minimum_required(VERSION 2.8.3)
project(sgdqn_common)

set(CMAKE_BUILD_TYPE "Release")
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-unused-variable -pthread")
else( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread -fopenmp")
endif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
find_package(catkin REQUIRED COMPONENTS
  message_filters
  message_generation
  roscpp
  rospy
  std_msgs
  geometry_msgs
  costmap_converter
  dynamic_reconfigure
)
#add dynamic reconfigure api
#find_package(catkin REQUIRED dynamic_reconfigure)
generate_dynamic_reconfigure_options(
  config/goal.cfg
)
add_message_files(DIRECTORY msgs FILES ObserveInfo.msg
RobotState.msg
PedState.msg
ActionCmd.msg
DiscState.msg
LineState.msg
)

add_service_files(DIRECTORY srv FILES TebCrowdSim.srv)

generate_messages(DEPENDENCIES std_msgs   geometry_msgs   costmap_converter)
set(sgdqn_common_includes
               ${catkin_INCLUDE_DIRS}
)
include_directories(${sgdqn_common_includes})
catkin_package(
 LIBRARIES custom_dynamic_reconfigure
  CATKIN_DEPENDS dynamic_reconfigure roscpp std_msgs)
  INCLUDE_DIRS ${sgdqn_common_includes}
)


