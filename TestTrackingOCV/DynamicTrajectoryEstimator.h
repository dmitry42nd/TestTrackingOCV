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

  double getScale(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers);

  void buildTrack(int frameIdF, int frameIdL);
  void loadOnlyDynamicsTracksFromFile(std::string &pathToAllTracks);

  void updateEstimates(int frameId);
  void finilizeTrack(Track& t);
  void registerTrack(Track& t);

  static void scaleSolver(std::vector<std::vector<cv::Point2d>> obs,
                   std::vector<std::vector<double>> cameras,
                   std::vector<cv::Mat> inliers,
                   std::vector<cv::Point3d> scaleXsF,
                   cv::Point3d Vest);


protected:
  char imgn[100];
  double scale_;
  const double essThr = 0.0005;
  const double pnpThr = 0.03;
  //for [255, 295]
  /*const double essThr = 0.01;
  const double pnpThr = 0.03;*/
  std::vector<std::shared_ptr<Track>> dynamicTracks;
  std::vector<cv::Point3d> objectPoints;
  std::vector<double *> objectPoints_ceres;
  std::vector<cv::Point2d> imagePoints;
  /*std::vector<cv::Point2d>  unPointsF, unPointsL;
  std::vector<std::vector<std::shared_ptr<TrackedPoint>>::iterator> its;*/

  std::vector<cv::Mat> owTs;

  CameraPoseProvider & poseProvider;
  cv::Mat const& K;
  cv::Mat const& dist;

  std::ofstream dataOut;
  std::ofstream dataOut_gt;
  std::ofstream errOut;
  cv::Mat img;
  std::vector<cv::Mat> oldrvec;
  std::vector<cv::Mat> oldtvec;
  histVector hists_;
  void filterByMaskDebug(cv::Mat const &mask, std::vector<cv::Point2d> &vF, std::vector<cv::Point2d> &vL,
                         std::vector<cv::Point2d> &v, histVector &its, std::vector<int> &trackIds,  int i);
  void getProjectionAndNormCeres(double *camera, double *point, cv::Point2f &pp, cv::Point3f &np);

  //std::vector<cv::Point2d> unPointsF, unPointsL;

  //void setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers, std::vector<cv::Point3d> &Xs);
  void setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers,std::vector<cv::Point3d> &Xs,std::vector<cv::Point2d> &projXs_debug);
  void block1(int frameIdF, int frameIdL);

  void reset();

  std::vector<std::vector<cv::Point2d>> scaleObs;
  std::vector<std::vector<double>> scaleCameras;
  std::vector<cv::Mat> scaleInliers;
  std::vector<cv::Point3d> scaleXsF;
  std::vector<cv::Point3d> scaleXsL;

  std::vector<cv::Point2d> projXsL_debug;

  int Fdebug_;
  int Ldebug_;
};

