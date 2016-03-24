//
// Created by dmitry on 3/23/16.
//

#include "ceres/rotation.h"

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, double *camera)
      : observed_x(observed_x), observed_y(observed_y), camera(camera) {
  }

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    double p[3];
    ceres::AngleAxisRotatePoint(camera, (double *)point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    double xp = - p[0] / p[2];
    double yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const double& l1 = camera[8];
    const double& l2 = camera[9];
    double r2 = xp*xp + yp*yp;
    double distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const double& focal_x = camera[6];
    const double& focal_y = camera[7];

    const double& shift_x = camera[10];
    const double& shift_y = camera[11];
    double predicted_x = focal_x * distortion * xp + shift_x;
    double predicted_y = focal_y * distortion * yp + shift_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y,
                                      double *camera) {
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 1, 3>(
                 new SnavelyReprojectionError(observed_x, observed_y, camera)));
   }

  double observed_x;
  double observed_y;
  double *camera;
};
