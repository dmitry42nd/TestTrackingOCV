cmake_minimum_required(VERSION 3.4)
project(TestTrackingOCV)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
  TestTrackingOCV/CameraPoseProvider.cpp
  TestTrackingOCV/CameraPoseProvider.h
  TestTrackingOCV/CameraPoseProviderTXT.cpp
  TestTrackingOCV/CameraPoseProviderTXT.h
  TestTrackingOCV/DynamicTrajectoryEstimator.cpp
  TestTrackingOCV/DynamicTrajectoryEstimator.h
  TestTrackingOCV/stdafx.cpp
  TestTrackingOCV/stdafx.h
  TestTrackingOCV/targetver.h
  TestTrackingOCV/TestTrackingOCV.cpp
  TestTrackingOCV/Track.cpp
  TestTrackingOCV/Track.h
  TestTrackingOCV/TrackedPoint.cpp
  TestTrackingOCV/TrackedPoint.h
  TestTrackingOCV/Tracker.cpp
  TestTrackingOCV/Tracker.h
  TestTrackingOCV/TrajectoryArchiver.cpp
  TestTrackingOCV/TrajectoryArchiver.h
  TestTrackingOCV/TriangulateError.h)

add_executable(TestTrackingOCV ${SOURCE_FILES})

#link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu/)
TARGET_LINK_LIBRARIES(TestTrackingOCV ${CERES_LIBRARIES})
TARGET_LINK_LIBRARIES(TestTrackingOCV
  opencv_highgui opencv_core opencv_calib3d opencv_features2d opencv_imgproc opencv_legacy opencv_video
  boost_filesystem boost_system ${EXTERNAL_LIBS})
