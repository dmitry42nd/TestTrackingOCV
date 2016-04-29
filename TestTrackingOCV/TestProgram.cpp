//
// Created by dmitry on 4/26/16.
//

#include <stdlib.h>

#include "TestProgram.h"
#include "DynamicTrajectoryEstimator.h"

void makeCamera(std::vector<double> & camera, cv::Vec3d t) {
  camera.push_back(0);
  camera.push_back(0);
  camera.push_back(0);
  camera.push_back(-t[0]);
  camera.push_back(-t[1]);
  camera.push_back(-t[2]);
}

static const double sc_coef = 0.2;
void TestProgram::rigid_body_kin() {
  srand((unsigned)time(NULL));

  pts[0] = cv::Vec3d(-1, -1, -1);
  pts[1] = cv::Vec3d(-1, -1,  1);
  pts[2] = cv::Vec3d(-1,  1, -1);
  pts[3] = cv::Vec3d(-1,  1,  1);
  pts[4] = cv::Vec3d( 1, -1, -1);
  pts[5] = cv::Vec3d( 1, -1,  1);
  pts[6] = cv::Vec3d( 1,  1, -1);
  pts[7] = cv::Vec3d( 1,  1,  1);

  cv::Mat R_object_ini;
  cv::Vec3d rvec_object_ini = cv::Vec3d( 0.1,  0.0,  0.2); //rand rotation
  cv::Rodrigues(rvec_object_ini, R_object_ini);
  for(auto & p : pts)
    p = cv::Vec3d(cv::Mat(R_object_ini*cv::Mat(p))) + cv::Vec3d(1.0, 0.0, 4.0); //"rand" translation

  //double data_rv[3] = {0.002, 0.00, 0.001};
  double data_rv[3] = {0., 0., 0.};
  cv::Mat rv = cv::Mat(1, 3, CV_64F, data_rv);
  cv::Rodrigues(rv, Rv);

  v_object = cv::Vec3d(0.0, 0.0, 0.1);
  a_object = cv::Vec3d(0.0, 0.0, 0.00);

  v_camera = cv::Vec3d(0, 0.009,  0.01);
  a_camera = cv::Vec3d(0,  0.01, -0.01);

  v_object_curr = v_object;
  v_camera_curr = v_camera;

  cur_pts = pts;
  /*for(auto i = 0; i < N; ++i) {
    std::cout << cur_pts[i] << std::endl;
  }
  std::cout << std::endl;*/

  campos = cv::Vec3d(0, 0, 0);

  std::vector<std::vector<cv::Point2d>> obs_;
  std::vector<std::vector<double>> cameras_;
  for(auto mom = 0; mom < TimeSpan; ++mom) {
    v_object_curr += a_object;
    v_camera_curr += a_camera;
    campos        += v_camera_curr;


    std::vector<double> camera;
    makeCamera(camera, campos);
    cameras_.push_back(camera);

    std::vector<cv::Point2d> projs_apt;
    for(auto j = 0; j < N; ++j) {
      //std::cout << cv::Vec3d(cv::Mat(Rv*cv::Mat(cur_pts[j]))) << " " << v_object_curr << std::endl;
      cur_pts[j]     = cv::Vec3d(cv::Mat(Rv*cv::Mat(cur_pts[j]))) + v_object_curr;
      cur_pts_cam[j] = cur_pts[j] - campos;

      double n0=((double)rand()/(double)RAND_MAX)*0.001 ;
      double n1=((double)rand()/(double)RAND_MAX)*0.001;
      cv::Vec3d tmp = cur_pts_cam[j]/cur_pts_cam[j][2] /* + 0.001*randn(2,1)*/;
      cv::Vec2d proj = cv::Vec2d(tmp[0]+n0, tmp[1]+n1);
      pt_projs[j].push_back(proj);
      projs_apt.push_back(proj);
    }
    obs_.push_back(projs_apt);
  }

  cv::Vec3d v_est = (cur_pts[0] - pts[0]) / TimeSpan;

  std::vector<cv::Point3d> XsF_;
  for(auto p : pts)
    XsF_.push_back(sc_coef*p);
  cv::Point3d V_ = cv::Point3d(sc_coef*v_est);
  std::vector<cv::Mat> inliers_; //not used

  DynamicTrajectoryEstimator::scaleSolver(obs_, cameras_, inliers_, XsF_, V_);
}