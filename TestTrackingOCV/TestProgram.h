//
// Created by dmitry on 4/26/16.
//

#ifndef TESTTRACKINGOCV_TESTPROGRAM_H
#define TESTTRACKINGOCV_TESTPROGRAM_H

#include "stdafx.h"

class TestProgram {
public:
  void rigid_body_kin();

protected:
  static const int N = 8;
  static const int TimeSpan = 100;

  std::array<cv::Vec3d, N> pts;
  std::array<cv::Vec3d, N> cur_pts;
  std::array<cv::Vec3d, N> cur_pts_cam;
  std::array<std::vector<cv::Vec2d>, N> pt_projs;

  cv::Mat Rv;// = rodrigues([0.001; 0; 0]);
  cv::Vec3d v_object;
  cv::Vec3d a_object;
  cv::Vec3d v_camera;
  cv::Vec3d a_camera;
  cv::Vec3d v_object_curr;
  cv::Vec3d v_camera_curr;

  cv::Vec3d campos;


};


#endif //TESTTRACKINGOCV_TESTPROGRAM_H
