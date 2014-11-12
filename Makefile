
OPENCV_PREFIX = /usr/pack/opencv-2.3.1-cr/amd64-debian-linux6.0
BOOST_PREFIX= /usr/pack/boost-1.47.0-bs/amd64-debian-linux6.0
INCLUDES = -I$(OPENCV_PREFIX)/include \
           -I$(BOOST_PREFIX)/include \
           -I/usr/include
LIBS = -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_core \
       -lboost_system -lboost_thread-mt -lboost_serialization -lboost_filesystem
LIBDIRS = -L$(OPENCV_PREFIX)/lib -Wl,-rpath,$(OPENCV_PREFIX)/lib \
          -L$(BOOST_PREFIX)/lib -Wl,-rpath,$(BOOST_PREFIX)/lib \
          -L/usr/lib
CC=g++
CFLAGS=-c -Wall -O3
SOURCES= image_sample.cpp head_pose_sample.cpp multi_part_sample.cpp face_forest.cpp face_utils.cpp demo.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=FaceForest

all: $(SOURCES) $(EXECUTABLE)
clean: 
	rm -f *.o
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@
