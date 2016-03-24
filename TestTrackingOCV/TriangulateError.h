//
// Created by dmitry on 3/23/16.
//

#include "ceres/rotation.h"

struct TriangulateError {
  TriangulateError(double observed_x, double observed_y, const double *camera)
      : observed_x(observed_x), observed_y(observed_y), camera(camera)
  { }

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    T rot[3];
    rot[0] = T(camera[0]);
    rot[1] = T(camera[1]);
    rot[2] = T(camera[2]);
    ceres::AngleAxisRotatePoint(rot, point, p);
    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]); p[1] += T(camera[4]); p[2] += T(camera[5]);

    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // The error is the difference between the predicted and observed position.
    residuals[0] = T(xp) - T(observed_x);
    residuals[1] = T(yp) - T(observed_y);

    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y,
                                      const double *camera) {
     return (new ceres::AutoDiffCostFunction<TriangulateError, 2, 3>(
                 new TriangulateError(observed_x, observed_y, camera)));
   }

  double observed_x;
  double observed_y;
  const double *camera;
};
