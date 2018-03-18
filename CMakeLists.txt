cmake_minimum_required(VERSION 2.8.12)
project(facial-feature-alignment)

add_subdirectory(/home/ajay/soft/dlib-19.9 dlib_build)

add_executable(face_tracker_images face_tracker_images.cpp)
target_link_libraries(face_tracker_images dlib::dlib)

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build face_tracker.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      find_package(Eigen3 QUIET)
      if (Eigen3_FOUND)
          add_executable(face_tracker face_tracker.cpp)
          target_link_libraries(face_tracker dlib::dlib ${OpenCV_LIBS} ${Eigen3_LIBS} )
      else()
          message("Eigen3 not found, so we won't build face_tracker.")
      endif()
   else()
      message("OpenCV not found, so we won't build face_tracker.")
   endif()
endif()

