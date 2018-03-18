// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include <iostream>

using namespace dlib;
using namespace std;

// Hold detected face shapes for this many frames.
constexpr int MAX_STALE_FACE_FRAMES = 3;
constexpr int OUTER_LIP_START = 49;
constexpr int NUM_OUTER_LIP_POINTS = 59 - OUTER_LIP_START + 2;


namespace {

    cv::Rect2i shape_bounding_box(full_object_detection shape, unsigned long start_index, unsigned long num_parts) {
        long tl_x = shape.part(start_index).x();
        long tl_y = shape.part(start_index).y();
        long br_x = shape.part(start_index).x();
        long br_y = shape.part(start_index).y();

        for (unsigned long i = start_index + 1; i < start_index + num_parts; ++i) {
            const auto& point = shape.part(i);
            tl_x = MIN(tl_x, point.x());
            tl_y = MIN(tl_y, point.y());
            br_x = MAX(br_x, point.x());
            br_y = MAX(br_y, point.y());
        }

        cv::Rect2i bbox(tl_x, tl_y, br_x - tl_x + 1, br_y - tl_y + 1);
        return bbox;
    }

    cv::Mat crop_to_polygon(cv::Mat source, full_object_detection shape, unsigned long start_index, unsigned long num_parts) {
        // crop source to polygon defined by points shape.parts(start_index) through shape.parts(end_index)

        const cv::Rect2i bbox = shape_bounding_box(shape, start_index, num_parts);
        cout << "Bounding box: " << bbox << endl;

        cv::Mat roi = source(bbox);

        // Translate shape points into frame
        std::vector<cv::Point2i> polygon(num_parts);
        for (unsigned long i = start_index; i < start_index + num_parts; ++i) {
            polygon.emplace_back(
                    (int) shape.part(i).x() - bbox.x,
                    (int) shape.part(i).y() - bbox.y);
        }

        // TODO: Mask out polygon
        cv::Mat mask = cv::Mat::zeros(roi.rows, roi.cols, CV_8U);
        cv::fillConvexPoly(mask, polygon.data(), num_parts, cv::Scalar(1));

        cv::Mat roi_masked;
        roi.copyTo(roi, roi_masked);

        return roi_masked;
    }

    void draw_shape(cv::Mat canvas, full_object_detection shape, unsigned long start_index, unsigned long num_parts) {
        for (unsigned long i = start_index; i < start_index + num_parts; ++i) {
            cv::Point2l center(shape.part(i).x(), shape.part(i).y());
            cv::circle(canvas, center, 3, cv::Scalar(0x00ff00));
        }
    }

} // end anonymous namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: face_tracker path/to/shape_predictor_68_face_landmarks.dat <camera_index> <reference_images...>" << endl;
    }

    int camera = stoi(argv[2]);

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize(argv[1]) >> pose_model;

    std::vector<cv::Mat> cvmouths;

    // Load image and extract face parts
    for (int i = 3; i < argc; ++i) {
        cout << "processing image " << argv[i] << endl;
        array2d<rgb_pixel> img;
        load_image(img, argv[i]);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        cv::Mat img_mat = toMat(img);

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        for (unsigned long j = 0; j < dets.size(); ++j) {
            full_object_detection shape = pose_model(img, dets[j]);
            cout << "number of parts: "<< shape.num_parts() << endl;
            cout << "pixel position of first part:  " << shape.part(0) << endl;
            cout << "pixel position of second part: " << shape.part(1) << endl;

            cv::Mat cvmouth = crop_to_polygon(img_mat, shape, OUTER_LIP_START, NUM_OUTER_LIP_POINTS);
            cv::cvtColor(cvmouth, cvmouth, cv::COLOR_BGR2RGB);
            cvmouths.emplace_back();
            cvmouth.copyTo(cvmouths.back());
        }
    }


    try {
        cv::VideoCapture cap(camera);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        std::vector<full_object_detection> shapes;
        int num_frames_no_detection = 0;

        // Grab and process frames until the main window is closed by the user.
        //image_window win;

        cv::namedWindow("camera canvas", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("matching mouth", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("dest", cv::WINDOW_AUTOSIZE);

        while(true) {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);

            // Find the pose of each face.
            cout << "Detected " << faces.size() << " faces" << endl;
            if (faces.size() > 0) {
                shapes.clear();
                num_frames_no_detection = 0;

                for (unsigned long i = 0; i < faces.size(); ++i) {
                    auto shape = pose_model(cimg, faces[i]);

                    //draw_shape(temp, shape, 0, shape.num_parts());
                    draw_shape(temp, shape, OUTER_LIP_START, NUM_OUTER_LIP_POINTS);

                    shapes.push_back(shape);

                    const cv::Rect2i dst_roi_bbox = shape_bounding_box(shape, OUTER_LIP_START, NUM_OUTER_LIP_POINTS);
                    cv::Mat dst_roi = temp(dst_roi_bbox);
                    cv::imshow("dest", dst_roi);

                    cv::Size dst_size(dst_roi.cols, dst_roi.rows);

                    const cv::Mat& matching_mouth = cvmouths.at(0);
                    cv::Mat matching_mouth_resized;
                    cv::resize(matching_mouth, matching_mouth_resized, dst_size);
                    matching_mouth_resized.copyTo(dst_roi);
                    cv::imshow("matching mouth", matching_mouth_resized);
                }

            } else if (num_frames_no_detection < MAX_STALE_FACE_FRAMES) {
                num_frames_no_detection++;
            } else {
                shapes.clear();
                num_frames_no_detection++;
            }

            // Display it all on the screen

            cv::imshow("camera canvas", temp);
#if 0
            win.clear_overlay();
            win.set_image(cimg);
            //win.add_overlay(render_face_detections(shapes));

            if (cvmouths.size() > 0) {
                cv_image<rgb_pixel> mouth(cvmouths.at(0));
                win.set_image(mouth);
            }
#endif

            cv::waitKey(10);
        }
    } catch(serialization_error& e) {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    } catch(exception& e) {
        cout << e.what() << endl;
    }
}

