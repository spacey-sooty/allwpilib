// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "fieldcalibration.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <frc/apriltag/AprilTagDetection.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <wpi/json.h>

#include "apriltag.h"
#include "gtsam_meme/wpical_gtsam.h"
#include "tag36h11.h"

struct CameraModel {
  Eigen::Matrix<double, 3, 3> intrinsic_matrix;
  Eigen::Matrix<double, 8, 1> distortion_coefficients;
};

inline CameraModel load_camera_model(std::string path) {
  Eigen::Matrix<double, 3, 3> camera_matrix;
  Eigen::Matrix<double, 8, 1> camera_distortion;

  std::ifstream file(path);

  wpi::json json_data;

  try {
    json_data = wpi::json::parse(file);
  } catch (const wpi::json::parse_error& e) {
    std::cout << e.what() << std::endl;
  }

  bool isCalibdb = json_data.contains("camera");

  if (!isCalibdb) {
    for (int i = 0; i < camera_matrix.rows(); i++) {
      for (int j = 0; j < camera_matrix.cols(); j++) {
        camera_matrix(i, j) =
            json_data["camera_matrix"][(i * camera_matrix.cols()) + j];
      }
    }

    for (int i = 0; i < camera_distortion.rows(); i++) {
      for (int j = 0; j < camera_distortion.cols(); j++) {
        camera_distortion(i, j) = json_data["distortion_coefficients"]
                                           [(i * camera_distortion.cols()) + j];
      }
    }
  } else {
    for (int i = 0; i < camera_matrix.rows(); i++) {
      for (int j = 0; j < camera_matrix.cols(); j++) {
        try {
          camera_matrix(i, j) = json_data["camera_matrix"][i][j];
        } catch (...) {
          camera_matrix(i, j) = json_data["camera_matrix"]["data"]
                                         [(i * camera_matrix.cols()) + j];
        }
      }
    }

    for (int i = 0; i < camera_distortion.rows(); i++) {
      for (int j = 0; j < camera_distortion.cols(); j++) {
        try {
          camera_distortion(i, j) =
              json_data["distortion_coefficients"]
                       [(i * camera_distortion.cols()) + j];
        } catch (...) {
          camera_distortion(i, j) = 0.0;
        }
      }
    }
  }

  cameracalibration::CameraModel camera_model{
      camera_matrix, camera_distortion, json_data["avg_reprojection_error"]};
  return camera_model;
}

inline cameracalibration::CameraModel load_camera_model(wpi::json json_data) {
  // Camera matrix
  Eigen::Matrix<double, 3, 3> camera_matrix;

  for (int i = 0; i < camera_matrix.rows(); i++) {
    for (int j = 0; j < camera_matrix.cols(); j++) {
      camera_matrix(i, j) =
          json_data["camera_matrix"][(i * camera_matrix.cols()) + j];
    }
  }

  // Distortion coefficients
  Eigen::Matrix<double, 8, 1> camera_distortion;

  for (int i = 0; i < camera_distortion.rows(); i++) {
    for (int j = 0; j < camera_distortion.cols(); j++) {
      camera_distortion(i, j) = json_data["distortion_coefficients"]
                                         [(i * camera_distortion.cols()) + j];
    }
  }

  cameracalibration::CameraModel camera_model{
      camera_matrix, camera_distortion, json_data["avg_reprojection_error"]};
  return camera_model;
}

inline std::map<int, wpi::json> load_ideal_map(std::string path) {
  std::ifstream file(path);
  wpi::json json_data = wpi::json::parse(file);
  std::map<int, wpi::json> ideal_map;

  for (const auto& element : json_data["tags"]) {
    ideal_map[element["ID"]] = element;
  }

  return ideal_map;
}

inline void draw_tag_cube(cv::Mat& frame,
                          Eigen::Matrix<double, 4, 4> camera_to_tag,
                          const Eigen::Matrix<double, 3, 3>& camera_matrix,
                          const Eigen::Matrix<double, 8, 1>& camera_distortion,
                          double tag_size) {
  cv::Mat camera_matrix_cv;
  cv::Mat camera_distortion_cv;

  cv::eigen2cv(camera_matrix, camera_matrix_cv);
  cv::eigen2cv(camera_distortion, camera_distortion_cv);

  std::vector<cv::Point3f> points_3d_box_base = {
      cv::Point3f(-tag_size / 2.0, tag_size / 2.0, 0.0),
      cv::Point3f(tag_size / 2.0, tag_size / 2.0, 0.0),
      cv::Point3f(tag_size / 2.0, -tag_size / 2.0, 0.0),
      cv::Point3f(-tag_size / 2.0, -tag_size / 2.0, 0.0)};

  std::vector<cv::Point3f> points_3d_box_top = {
      cv::Point3f(-tag_size / 2.0, tag_size / 2.0, -tag_size),
      cv::Point3f(tag_size / 2.0, tag_size / 2.0, -tag_size),
      cv::Point3f(tag_size / 2.0, -tag_size / 2.0, -tag_size),
      cv::Point3f(-tag_size / 2.0, -tag_size / 2.0, -tag_size)};

  std::vector<cv::Point2f> points_2d_box_base = {
      cv::Point2f(0.0, 0.0), cv::Point2f(0.0, 0.0), cv::Point2f(0.0, 0.0),
      cv::Point2f(0.0, 0.0)};

  std::vector<cv::Point2f> points_2d_box_top = {
      cv::Point2f(0.0, 0.0), cv::Point2f(0.0, 0.0), cv::Point2f(0.0, 0.0),
      cv::Point2f(0.0, 0.0)};

  Eigen::Matrix<double, 3, 3> r_vec = camera_to_tag.block<3, 3>(0, 0);
  Eigen::Matrix<double, 3, 1> t_vec = camera_to_tag.block<3, 1>(0, 3);

  cv::Mat r_vec_cv;
  cv::Mat t_vec_cv;

  cv::eigen2cv(r_vec, r_vec_cv);
  cv::eigen2cv(t_vec, t_vec_cv);

  cv::projectPoints(points_3d_box_base, r_vec_cv, t_vec_cv, camera_matrix_cv,
                    camera_distortion_cv, points_2d_box_base);
  cv::projectPoints(points_3d_box_top, r_vec_cv, t_vec_cv, camera_matrix_cv,
                    camera_distortion_cv, points_2d_box_top);

  for (int i = 0; i < 4; i++) {
    cv::Point2f& point_base = points_2d_box_base.at(i);
    cv::Point2f& point_top = points_2d_box_top.at(i);

    cv::line(frame, point_base, point_top, cv::Scalar(0, 255, 255), 5);

    int i_next = (i + 1) % 4;
    cv::Point2f& point_base_next = points_2d_box_base.at(i_next);
    cv::Point2f& point_top_next = points_2d_box_top.at(i_next);

    cv::line(frame, point_base, point_base_next, cv::Scalar(0, 255, 255), 5);
    cv::line(frame, point_top, point_top_next, cv::Scalar(0, 255, 255), 5);
  }
}

/**
 * Convert a video file to a list of keyframes
 */
inline bool process_video_file(
    apriltag_detector_t* tag_detector,
    const Eigen::Matrix<double, 3, 3>& camera_matrix,
    const Eigen::Matrix<double, 8, 1>& camera_distortion, double tag_size,
    const std::string& path, gtsam::Key& keyframe,
    wpical::KeyframeMap& outputMap, bool show_debug_window) {
  // clear inputs
  outputMap.clear();

  if (show_debug_window) {
    cv::namedWindow("Processing Frame", cv::WINDOW_NORMAL);
  }
  cv::VideoCapture video_input(path);

  if (!video_input.isOpened()) {
    std::cout << "Unable to open video " << path << std::endl;
    return false;
  }

  // Reuse mats if we can - allocatiosn are expensive
  cv::Mat frame;
  cv::Mat frame_gray;
  cv::Mat frame_debug;

  while (video_input.read(frame)) {
    std::cout << "Processing " << path << " - Frame " << keyframe << std::endl;

    // Convert color frame to grayscale frame
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // Clone color frame for debugging
    frame_debug = frame.clone();

    // Detect tags
    image_u8_t tag_image = {frame_gray.cols, frame_gray.rows, frame_gray.cols,
                            frame_gray.data};
    zarray_t* tag_detections =
        apriltag_detector_detect(tag_detector, &tag_image);

    // Skip this loop if there are no tags detected
    if (zarray_size(tag_detections) == 0) {
      std::cout << "No tags detected" << std::endl;
      continue;
    }

    std::vector<TagDetection> tagsThisKeyframe;
    for (int i = 0; i < zarray_size(tag_detections); i++) {
      apriltag_detection_t* tag_detection_i;
      zarray_get(tag_detections, i, &tag_detection_i);

      // Convert to our data type. I don't love how complicated this is.
      auto atag = reinterpret_cast<frc::AprilTagDetection*>(tag_detection_i);
      auto tag_corners_cv = std::vector<cv::Point2d>{};
      for (int corn = 0; corn < 4; corn++) {
        tag_corners_cv.emplace_back(atag->GetCorner(corn).x,
                                    atag->GetCorner(corn).y);
      }

      // Undistort so gtsam doesn't have to deal with distortion
      cv::Mat camCalCv(camera_matrix.rows(), camera_matrix.cols(), CV_64F);
      cv::Mat camDistCv(camera_distortion.rows(), camera_distortion.cols(),
                        CV_64F);

      cv::eigen2cv(camera_matrix, camCalCv);
      cv::eigen2cv(camera_distortion, camDistCv);

      cv::undistortImagePoints(tag_corners_cv, tag_corners_cv, camCalCv,
                               camDistCv);

      TagDetection tag;
      tag.id = atag->GetId();

      tag.corners.resize(4);
      std::transform(tag_corners_cv.begin(), tag_corners_cv.end(),
                     tag.corners.begin(),
                     [](const auto& it) { return TargetCorner{it.x, it.y}; });

      tagsThisKeyframe.push_back(tag);
    }
    outputMap[keyframe] = tagsThisKeyframe;

    apriltag_detections_destroy(tag_detections);

    // Show debug
    if (show_debug_window) {
      cv::imshow("Processing Frame", frame_debug);
      cv::waitKey(1);
    }

    // Keep track of the frame number
    keyframe++;
  }

  video_input.release();
  if (show_debug_window) {
    cv::destroyAllWindows();
  }

  return true;
}

int fieldcalibration::calibrate(std::string input_dir_path,
                                std::string output_file_path,
                                std::string camera_model_path,
                                std::string ideal_map_path, int pinned_tag_id,
                                bool show_debug_window) {
  // // Silence OpenCV logging
  // cv::utils::logging::setLogLevel(
  //     cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

  // Load camera model
  Eigen::Matrix3d camera_matrix;
  Eigen::Matrix<double, 8, 1> camera_distortion;

  try {
    auto camera_model = load_camera_model(camera_model_path);
    camera_matrix = camera_model.intrinsic_matrix;
    camera_distortion = camera_model.distortion_coefficients;
  } catch (...) {
    return 1;
  }

  // Convert intrinsics to gtsam-land. Order fx fy s u0 v0
  gtsam::Cal3_S2 gtsam_cal{
      camera_matrix(0, 0), camera_matrix(1, 1), 0,
      camera_matrix(0, 2), camera_matrix(1, 2),
  };

  wpi::json json = wpi::json::parse(std::ifstream(ideal_map_path));
  if (!json.contains("tags")) {
    return 1;
  }

  // Load ideal field map
  std::map<int, wpi::json> ideal_map;
  try {
    ideal_map = load_ideal_map(ideal_map_path);
  } catch (...) {
    return 1;
  }

  // Apriltag detector
  apriltag_detector_t* tag_detector = apriltag_detector_create();
  tag_detector->nthreads = 8;

  apriltag_family_t* tag_family = tag36h11_create();
  apriltag_detector_add_family(tag_detector, tag_family);

  // Write down keyframes from all our video files
  wpical::KeyframeMap outputMap;

  gtsam::Key keyframe{gtsam::symbol_shorthand::X(0)};

  constexpr units::meter_t TAG_SIZE = 0.1651_m;

  for (const auto& entry :
       std::filesystem::directory_iterator(input_dir_path)) {
    // Ignore hidden files
    if (entry.path().filename().string()[0] == '.') {
      continue;
    }

    const std::string path = entry.path().string();

    bool success = process_video_file(
        tag_detector, camera_matrix, camera_distortion, TAG_SIZE.to<double>(),
        path, keyframe, outputMap, show_debug_window);

    if (!success) {
      std::cout << "Unable to process video " << path << std::endl;
      return 1;
    }
  }

<<<<<<< HEAD
  // Build optimization problem
  ceres::Problem problem;
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses.find(constraint.id_begin);
    auto pose_end_iter = poses.find(constraint.id_end);

    ceres::CostFunction* cost_function =
        PoseGraphError::Create(constraint.t_begin_end);

    problem.AddResidualBlock(cost_function, nullptr,
                             pose_begin_iter->second.p.data(),
                             pose_begin_iter->second.q.coeffs().data(),
                             pose_end_iter->second.p.data(),
                             pose_end_iter->second.q.coeffs().data());

    problem.SetManifold(pose_begin_iter->second.q.coeffs().data(),
                        quaternion_manifold);
    problem.SetManifold(pose_end_iter->second.q.coeffs().data(),
                        quaternion_manifold);
  }

  // Pin tag
  auto pinned_tag_iter = poses.find(pinned_tag_id);
  if (pinned_tag_iter != poses.end()) {
    problem.SetParameterBlockConstant(pinned_tag_iter->second.p.data());
    problem.SetParameterBlockConstant(
        pinned_tag_iter->second.q.coeffs().data());
  }

  // Solve
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.num_threads = 10;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  // Output
  std::map<int, wpi::json> observed_map = ideal_map;

  Eigen::Matrix<double, 4, 4> correction_a;
  correction_a << 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1;

  Eigen::Matrix<double, 4, 4> correction_b;
  correction_b << 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 1;

  Eigen::Matrix<double, 4, 4> pinned_tag_transform =
      get_tag_transform(ideal_map, pinned_tag_id);

  for (const auto& [tag_id, pose] : poses) {
    // Transformation from pinned tag
    Eigen::Matrix<double, 4, 4> transform =
        Eigen::Matrix<double, 4, 4>::Identity();

    transform.block<3, 3>(0, 0) = pose.q.toRotationMatrix();
    transform.block<3, 1>(0, 3) = pose.p;

    // Transformation from world
    Eigen::Matrix<double, 4, 4> corrected_transform =
        pinned_tag_transform * correction_a * transform * correction_b;
    Eigen::Quaternion<double> corrected_transform_q(
        corrected_transform.block<3, 3>(0, 0));

    observed_map[tag_id]["pose"]["translation"]["x"] =
        corrected_transform(0, 3);
    observed_map[tag_id]["pose"]["translation"]["y"] =
        corrected_transform(1, 3);
    observed_map[tag_id]["pose"]["translation"]["z"] =
        corrected_transform(2, 3);

    observed_map[tag_id]["pose"]["rotation"]["quaternion"]["X"] =
        corrected_transform_q.x();
    observed_map[tag_id]["pose"]["rotation"]["quaternion"]["Y"] =
        corrected_transform_q.y();
    observed_map[tag_id]["pose"]["rotation"]["quaternion"]["Z"] =
        corrected_transform_q.z();
    observed_map[tag_id]["pose"]["rotation"]["quaternion"]["W"] =
        corrected_transform_q.w();
  }

  wpi::json observed_map_json;

  for (const auto& [tag_id, tag_json] : observed_map) {
    observed_map_json["tags"].push_back(tag_json);
  }
=======
  // TODO - fill these in
  wpical::GtsamApriltagMap layoutGuess{{}, TAG_SIZE};
  std::map<int32_t, std::pair<gtsam::Pose3, gtsam::SharedNoiseModel>> fixedTags;

  // one pixel in u and v - TODO don't hardcode this
  gtsam::SharedNoiseModel cameraNoise{
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0)};

  auto calResult = wpical::OptimizeLayout(layoutGuess, outputMap, gtsam_cal,
                                          fixedTags, cameraNoise);

  wpi::json observed_map_json;

  // TODO add my tags to the output map
>>>>>>> c661eb25a (Hack upfieldcalibration)

  observed_map_json["field"] = {
      {"length", static_cast<double>(json.at("field").at("length"))},
      {"width", static_cast<double>(json.at("field").at("width"))}};

  std::ofstream output_file(output_file_path);
  output_file << observed_map_json.dump(4) << std::endl;

  return 0;
}
