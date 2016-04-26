#pragma once
#include "CameraPoseProvider.h"
#include "TrajectoryArchiver.h"
#include "Tracker.h"
#include "Track.h"

typedef std::vector<std::vector<std::shared_ptr<TrackedPoint>>> histVector;
typedef std::vector<std::vector<std::shared_ptr<TrackedPoint>>::const_iterator> histIterVector;

struct scaleProblemPack{
  double * camera;
  double * obs;
  double * v;
  double * X;
  double k;
};

class DynamicTrajectoryEstimator
{
public:
  DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider);

  void buildTrack(int frameIdF, int frameIdL);
  void loadOnlyDynamicsTracksFromFile(std::string &pathToAllTracks);

  void updateEstimates(int frameId);
  void finilizeTrack(Track& t);
  void registerTrack(Track& t);

protected:

  std::vector<std::shared_ptr<Track>> dynamicTracks;
  std::vector<cv::Point3d> objectPoints;
  std::vector<double *> objectPoints_ceres;
  std::vector<cv::Point2d> imagePoints;
  /*std::vector<cv::Point2d>  unPointsF, unPointsL;
  std::vector<std::vector<std::shared_ptr<TrackedPoint>>::iterator> its;*/

  std::vector<cv::Mat> owTs;
  std::vector<std::vector<cv::Mat>> oXs;

  CameraPoseProvider & poseProvider;
  cv::Mat const& K;
  cv::Mat const& dist;

  std::ofstream dataOut;
  std::ofstream errOut;
  cv::Mat img;
  cv::Mat inliers;
  std::vector<cv::Mat> oldrvec;
  std::vector<cv::Mat> oldtvec;
  histVector hists_;
  void filterByMaskDebug(cv::Mat const &mask, std::vector<cv::Point2d> &vF, std::vector<cv::Point2d> &vL,
                         std::vector<cv::Point2d> &v, histVector &its, int i);
  void getProjectionAndNormCeres(double *camera, double *point, cv::Point2f &pp, cv::Point3f &np);

  void scaleSolver(std::vector<std::vector<cv::Point2d>> obs,
       std::vector<std::vector<double>> cameras,
       std::vector<cv::Mat> inliers,
       std::vector<cv::Point3d> XsF , std::vector<cv::Point3d> XsL);

  //std::vector<cv::Point2d> unPointsF, unPointsL;

  //void setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers, std::vector<cv::Point3d> &Xs);
  void setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers,std::vector<cv::Point3d> &Xs,std::vector<cv::Point2d> &projXs_debug);
  void block1(int frameIdF, int frameIdL);

  std::vector<std::vector<cv::Point2d>> scaleObs;
  std::vector<std::vector<double>> scaleCameras;
  std::vector<cv::Mat> scaleInliers;
  std::vector<cv::Point3d> scaleXsF;
  std::vector<cv::Point3d> scaleXsL;

  std::vector<cv::Point2d> projXsL_debug;

  int Fdebug_;
  int Ldebug_;
};

