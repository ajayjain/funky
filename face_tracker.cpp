/*
    Note that the face detector is fastest when compiled with at least
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

#include <iostream>
#include <unordered_map>
#include <vector>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/geometry/point_transforms.h>
#include <dlib/geometry.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <eigen3/Eigen/Core>

using namespace dlib;
using namespace std;

// Hold detected face shapes for this many frames.
constexpr int MAX_STALE_FACE_FRAMES = 3;
constexpr int NUM_LANDMARKS = 68;
constexpr int OUTER_LIP_START = 49;
constexpr int NUM_OUTER_LIP_POINTS = 59 - OUTER_LIP_START + 2;
constexpr int NUM_LIP_POINTS = 67 - OUTER_LIP_START + 1;

typedef Eigen::MatrixX2i LandmarkMatrix;

template<class T>
using LandmarkVector = std::vector<dlib::vector<T, 2>>;

//typedef std::vector<dlib::vector<int, 2>> LandmarkVector;

namespace {

    cv::Rect2i shape_bounding_box(full_object_detection shape, int start_index, int num_parts,
                                  int max_y = 0, int max_x = 0, float padding = 0) {
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

        tl_x = MAX(tl_x - 50 * padding, 0);
        tl_y = MAX(tl_y - 50 * padding, 0);

        br_x = MIN(br_x + 50 * padding, max_x);
        br_y = MIN(br_y + 50 * padding, max_y);

        long width = max_x > tl_x ? MIN(max_x - tl_x, br_x - tl_x + 1) : br_x - tl_x + 1;
        long height = max_y > tl_y ? MIN(max_y - tl_y, br_y - tl_y + 1) : br_y - tl_y + 1;

        cv::Rect2i bbox(tl_x,
                        tl_y,
                        width,
                        height);

        return bbox;
    }

    LandmarkMatrix shape_to_points(full_object_detection shape) {
        LandmarkMatrix points(shape.num_parts(), 2);

        for (unsigned long i = 0; i < shape.num_parts(); ++i) {
            points(i, 0) = shape.part(i).x();
            points(i, 1) = shape.part(i).y();
        }

        return points;
    }

    template<class T>
    LandmarkVector<T> LandmarkMatrix_to_LandmarkVector(LandmarkMatrix pts, int start_index, int num_parts) {
        LandmarkVector<T> vec;

        for (int r = start_index; r < start_index + num_parts; ++r) {
            dlib::vector<T, 2> pt;
            pt(0) = pts(r, 0);
            pt(1) = pts(r, 1);
            vec.push_back(pt);
        }

        return vec;
    }

    void transform_by_points(LandmarkMatrix reference_pts, LandmarkMatrix source_pts, const cv::Mat source, cv::Mat dest, const int start_index, const int num_parts) {
        const cv_image<bgr_pixel> dlibsource(source);
        cv_image<bgr_pixel> dlibdestination(dest);

        const LandmarkVector<double> from_pts = LandmarkMatrix_to_LandmarkVector<double>(reference_pts, start_index, num_parts);
        const LandmarkVector<double> to_pts = LandmarkMatrix_to_LandmarkVector<double>(source_pts, start_index, num_parts);

        //cout << "From pts: " << from_pts.at(0) << ", " << from_pts.at(1) << endl;
        //cout << "To pts:   " << to_pts.at(0) << ", " << to_pts.at(1) << endl;

        const dlib::point_transform_projective trans = dlib::find_projective_transform(from_pts, to_pts);

#if 0
        dlib::transform_image(dlibsource,
                              dlibdestination,
                              dlib::interpolate_quadratic(),
                              trans,
                              dlib::no_background());
#else
        const int l = reference_pts.col(0).minCoeff();
        const int r = reference_pts.col(0).maxCoeff();
        const int t = reference_pts.col(1).minCoeff();
        const int b = reference_pts.col(1).maxCoeff();
        dlib::rectangle dest_area(l, t, r, b);

        dlib::transform_image(dlibsource,
                              dlibdestination,
                              dlib::interpolate_quadratic(),
                              trans,
                              dlib::no_background(),
                              dest_area);
#endif

        //return dlib::toMat(dlibdestination);
    }

    float landmark_deviation(LandmarkMatrix a, LandmarkMatrix b, int start_index, int num_parts) {
        LandmarkVector<int> from_pts = LandmarkMatrix_to_LandmarkVector<int>(a, start_index, num_parts);
        LandmarkVector<int> to_pts = LandmarkMatrix_to_LandmarkVector<int>(b, start_index, num_parts);

        dlib::point_transform_affine transform = dlib::find_similarity_transform(from_pts, to_pts);

        float mse = 0.0;

        for (int i = 0; i < from_pts.size(); ++i) {
            auto from_trans = transform(from_pts.at(i));

            auto diff = from_trans - to_pts.at(i);
            //double x_dev = from_trans(0) - to_pts(i, 0);
            //double y_dev = from_trans(1) - to_pts(i, 1);

            mse += diff(0) * diff(0) + diff(1) * diff(1);
            // mse += x_dev * x_dev + y_dev * y_dev;
        }

        mse /= num_parts;

        return mse;
    }

    std::pair<LandmarkMatrix, cv::Mat> crop_to_polygon(cv::Mat source, full_object_detection shape, int start_index, int num_parts, float padding) {
        // crop source to polygon defined by points shape.parts(start_index) through shape.parts(end_index)


#if 0
        // Translate shape points into frame
        std::vector<cv::Point2i> polygon(num_parts);
        for (unsigned long i = start_index; i < start_index + num_parts; ++i) {
            polygon.emplace_back(
                    (int) shape.part(i).x() - bbox.x,
                    (int) shape.part(i).y() - bbox.y);
        }

        // Copy only pixels within polygon
        cv::Mat mask = cv::Mat::zeros(roi.rows, roi.cols, CV_8U);
        cv::fillConvexPoly(mask, polygon.data(), num_parts, cv::Scalar(255));
        cv::Mat roi_masked;
        roi.copyTo(roi_masked, mask);

        cv::namedWindow("mask", cv::WINDOW_AUTOSIZE);
        cv::imshow("mask", mask);

        dlib::sleep(2000);

        return roi_masked;
#elif 0
        cv::Rect2i bbox = shape_bounding_box(shape, start_index, num_parts, source.rows, source.cols, padding);
        cout << "Bounding box: " << bbox << endl;

        cv::Mat roi = source(bbox);

        LandmarkMatrix polygon(shape.num_parts(), 2);
        for (int i = 0; i < shape.num_parts(); ++i) {
            polygon(i, 0) = shape.part(i).x() - bbox.x;
            polygon(i, 1) = shape.part(i).y() - bbox.y;
        }

        std::pair<LandmarkMatrix, cv::Mat> landmarks_and_roi(polygon, roi);
        return landmarks_and_roi;
#else
        // Create vector of points for hull
        // Find the center of the polygon, so we can displace bounds outward to add padding
        cv::Point2i polygon[num_parts];
        //cv::Point2i polygon_center(0, 0);
        for (int i = 0; i < num_parts; ++i) {
            int x = shape.part(start_index + i).x();
            int y = shape.part(start_index + i).y();

            polygon[i].x = x;
            polygon[i].y = y;

            //polygon_center.x += x;
            //polygon_center.y += y;
        }
        //polygon_center.x /= (float) num_parts;
        //polygon_center.y /= (float) num_parts;

        // Displace polygon vertices outward
        //for (int i = 0; i < num_parts; ++i) {
        //    auto vertex = polygon.at(i);
        //    auto displacement = vertex - polygon_center;
        //    polygon[i] = vertex + displacement * padding;
        //}

        // Copy only pixels within polygon
        cv::Mat mask = cv::Mat::zeros(source.rows, source.cols, CV_8UC1);
        cv::fillConvexPoly(mask, polygon, num_parts, 255, 8, 0);
        cv::Mat source_masked;
        source.copyTo(source_masked, mask);

        auto landmarks = shape_to_points(shape);
        std::pair<LandmarkMatrix, cv::Mat> landmarks_and_roi(landmarks, source_masked);
        return landmarks_and_roi;
#endif
    }

    void draw_shape(cv::Mat canvas, full_object_detection shape, unsigned long start_index, unsigned long num_parts) {
        for (unsigned long i = start_index; i < start_index + num_parts; ++i) {
            cv::Point2l center(shape.part(i).x(), shape.part(i).y());
            cv::circle(canvas, center, 3, cv::Scalar(0x00ff00));
        }
    }

    cv::Mat blend(const cv::Mat overlay, const cv::Mat source) {
        assert(overlay.rows == source.rows && overlay.cols == source.cols);

#if 1
        return cv::max(overlay, source);
#else
        cv::Mat out(source.rows, source.cols, source.type());
        for (int r = 0; r < overlay.rows; ++r) {
            for (int c = 0; c < overlay.cols; ++c) {
                //if (overlay.at<int>(r, c) > 0) {
                //    out.at<int>(r, c) = overlay.at<int>(r, c) * source.at<int>(r, c);
                //} else {
                //    out.at<int>(r, c) = overlay.at<int>(r, c);
                //}
            }
        }
        return out;
#endif
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

    std::vector<cv::Mat> cvmouths_mem; // Buffer where we can store matrices such that they won't be deallocated after
                                       // locals go out of scope in the loops below
    std::vector<std::pair<LandmarkMatrix, cv::Mat>> cvmouths;

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

            // Convert face shape into X by 2 matrix
            cvmouths_mem.emplace_back();

            std::pair<LandmarkMatrix, cv::Mat> cvmouth = crop_to_polygon(img_mat, shape, OUTER_LIP_START, NUM_OUTER_LIP_POINTS, 1.0);
            cv::cvtColor(cvmouth.second, cvmouths_mem.back(), cv::COLOR_BGR2RGB);

            cvmouths.emplace_back(cvmouth.first, cvmouths_mem.back());
        }
    }


    try {
        cv::VideoCapture cap(camera);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        std::vector<Eigen::MatrixX2i> shapes;
        int num_frames_no_detection = 0;

        cv::namedWindow("camera canvas", cv::WINDOW_AUTOSIZE);
        //cv::namedWindow("matching mouth", cv::WINDOW_AUTOSIZE);
        //cv::namedWindow("matching mouth transformed", cv::WINDOW_AUTOSIZE);

        // Grab and process frames in a loop.
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
                    // draw_shape(temp, shape, OUTER_LIP_START, NUM_LIP_POINTS);

                    auto landmarks = shape_to_points(shape);
                    shapes.push_back(landmarks);

                    int best_candidate = -1;
                    float lowest_mse = 999999;

                    for (int i = 0; i < cvmouths.size(); ++i) {
                        float mse = landmark_deviation(cvmouths.at(i).first, landmarks, OUTER_LIP_START, NUM_LIP_POINTS);

                        if (mse < lowest_mse) {
                            lowest_mse = mse;
                            best_candidate = i;
                        }

                        cout << "Deviation: " << mse << endl;
                    }

                    if (best_candidate >= 0) {
                        const std::pair<LandmarkMatrix, cv::Mat> &matching_mouth = cvmouths.at(best_candidate);

#if 0
                        const cv::Rect2i dst_roi_bbox = shape_bounding_box(shape, OUTER_LIP_START, NUM_LIP_POINTS, temp.rows, temp.cols);
                        cv::Mat dst_roi = temp(dst_roi_bbox);
                        cv::Size dst_size(dst_roi.cols, dst_roi.rows);
                        cv::imshow("dest", dst_roi);

                        cv::Mat matching_mouth_resized;
                        cv::resize(matching_mouth.second, matching_mouth_resized, dst_size);
                        matching_mouth_resized.copyTo(dst_roi);
                        cv::imshow("matching mouth", matching_mouth_resized);
#else
                        //cv::imshow("matching mouth", matching_mouth.second);


                        cv::Mat matching_mouth_transformed = cv::Mat::zeros(temp.rows, temp.cols, temp.type());
                        transform_by_points(landmarks, matching_mouth.first, matching_mouth.second, matching_mouth_transformed,
                                            OUTER_LIP_START, NUM_LIP_POINTS);
                        temp = blend(matching_mouth_transformed, temp);
                        // temp = matching_mouth_transformed;
                        //cv::imshow("matching mouth transformed", matching_mouth_transformed);
#endif
                    } else {
                        cout << "No candidate face parts!" << endl;
                    }
                }

            } else if (num_frames_no_detection < MAX_STALE_FACE_FRAMES) {
                num_frames_no_detection++;
            } else {
                shapes.clear();
                num_frames_no_detection++;
            }

            // Display it all on the screen

            cv::imshow("camera canvas", temp);
            cv::waitKey(20);
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

