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
    T p[3];

    //very important part!
    T rot[3];
    rot[0] = T(camera[0]);
    rot[1] = T(camera[1]);
    rot[2] = T(camera[2]);
    ceres::AngleAxisRotatePoint(rot, point, p);

    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

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

#if 1
struct TriangulateError2 {
  TriangulateError2(const double ox, const double oy, const double *c)
      : ox(ox), oy(oy), c(c)
  { }

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {

    T p[3];

    //very important part!
    T rot[3];
    rot[0] = T(c[0]);
    rot[1] = T(c[1]);
    rot[2] = T(c[2]);
    ceres::AngleAxisRotatePoint(rot, point, p);

    p[0] += T(c[3]);
    p[1] += T(c[4]);
    p[2] += T(c[5]);

    T xp = p[0] / p[2]; //no minus because of recoverPose()?
    T yp = p[1] / p[2];

    residuals[0] = T(xp) - T(ox);
    residuals[1] = T(yp) - T(oy);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double ox,
                                     const double oy,
                                     const double *c) {
    return (new ceres::AutoDiffCostFunction<TriangulateError2, 2, 3>(
        new TriangulateError2(ox, oy, c)));
  }

  double ox;
  double oy;
  const double *c;
};
#endif


struct TriangulateError3 {
  TriangulateError3(const double ox, const double oy)
      : ox(ox), oy(oy)
  { }

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {

    T p[3];

    ceres::AngleAxisRotatePoint(camera, point, p);

    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    T xpF = p[0] / p[2];
    T ypF = p[1] / p[2];

    residuals[0] = T(xpF) - T(ox);
    residuals[1] = T(ypF) - T(oy);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double ox,
                                     const double oy) {
    return (new ceres::AutoDiffCostFunction<TriangulateError3, 2, 6, 3>(
        new TriangulateError3(ox, oy)));
   }

  double ox;
  double oy;
};


struct ScaleError {
  ScaleError(const double obs_x,
             const double obs_y,
             const double X_x,
             const double X_y,
             const double X_z,
             const double *camera,
             const double k)
      : obs_x(obs_x), obs_y(obs_y), X_x(X_x), X_y(X_y), X_z(X_z), camera(camera), k(k)
  { }

  template <typename T>
  bool operator()(const T* const s,
                  /*const T* const R,*/
                  const T* const v,
                        T* residuals) const {
    T p[3], p_[3], point[3], point_[3];

    /*point_[0] = T(X_x);
    point_[1] = T(X_y);
    point_[2] = T(X_z);

    for(int i = 0; i < (int)k; ++i)
      ceres::AngleAxisRotatePoint(R, point_, p_);*/
#if 1
    /*point[0] = T(s[0])*(p_[0] + T(k)*T(v[0]));
    point[1] = T(s[0])*(p_[1] + T(k)*T(v[1]));
    point[2] = T(s[0])*(p_[2] + T(k)*T(v[2]));*/

    point[0] = T(s[0])*T(X_x) + T(k)*T(v[0]);
    point[1] = T(s[0])*T(X_y) + T(k)*T(v[1]);
    point[2] = T(s[0])*T(X_z) + T(k)*T(v[2]);
#else
    point[0] = T(X_x) + T(k)*T(v[0]);
    point[1] = T(X_y) + T(k)*T(v[1]);
    point[2] = T(X_z) + T(k)*T(v[2]);
#endif


    //very important part!
    T rot[3];
    rot[0] = T(camera[0]);
    rot[1] = T(camera[1]);
    rot[2] = T(camera[2]);
    ceres::AngleAxisRotatePoint(rot, point, p);

    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // The error is the difference between the predicted and observed position.
    residuals[0] = T(xp) - T(obs_x);
    residuals[1] = T(yp) - T(obs_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double obs_x,
                                     const double obs_y,
                                     const double X_x,
                                     const double X_y,
                                     const double X_z,
                                     const double *camera,
                                     const double k) {
    return (new ceres::AutoDiffCostFunction<ScaleError, 2, 1,/* 3,*/ 3>(
        new ScaleError(obs_x, obs_y, X_x, X_y, X_z, camera, k)));
  }

  const double k;
  const double obs_x, obs_y;  //2
  const double X_x, X_y, X_z; //3
  const double *camera;       //3
};