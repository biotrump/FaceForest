cmake_minimum_required(VERSION 2.8  FATAL_ERROR)
PROJECT( RTFFD-2012 )

#set variable from shell's env var by $ENV{env_var}
#include(cmake/CVEnvImport.cmake)

#opencv should be built first!
set(OpenCV_DIR $ENV{OPENCV_OUT})
#message("++++OpenCV_DIR:""${OpenCV_DIR}")
FIND_PACKAGE( OpenCV REQUIRED)
if(OpenCV_FOUND)
	message("found")
	message( "OpenCV_INCLUDE_DIRS:" "${OpenCV_INCLUDE_DIRS}" )
	message( "OpenCV_INCLUDES:" "${OpenCV_INCLUDES}")
	message( "OpenCV_LIBRARIES:" "${OpenCV_LIBRARIES}")
	message( "OpenCV_LIBS:" "${OpenCV_LIBS}")
	message( "OpenCV_DEFINITIONS:" "${OpenCV_DEFINITIONS}")
else()
	message(FATAL_ERROR "openCV not foud")
endif()

#openMP to avoid link error : omp_xxx error!
# undefined reference to `GOMP_loop_dynamic_start
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.49.0" "1.50.0" "1.51.0" "1.52.0" "1.53.0" "1.54.0" "1.55.0")
find_package(Boost 1.49
    COMPONENTS
      program_options
      unit_test_framework
      random
      serialization
      filesystem
      thread
      system
    REQUIRED
)
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})

# On Windows, automatic linking is performed, so we don't need to worry about
# it.  Clear the list of libraries to link against and let Visual Studio handle
# it.
if (WIN32)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
endif (WIN32)

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.
add_definitions(-DBOOST_TEST_DYN_LINK)

#	train_ffd.cpp
#train_headpose.cpp
#eval_ffd.cpp
ADD_EXECUTABLE( FaceForest
	demo.cpp
	face_forest.cpp
	face_utils.cpp
	head_pose_sample.cpp
	image_sample.cpp
	multi_part_sample.cpp
	)
#message(">>> DSP_ICA_LIBS::${DSP_ICA_LIBS}" )
#dynamic or static link
TARGET_LINK_LIBRARIES( FaceForest
	${OpenCV_LIBS}
	${Boost_LIBRARIES})
