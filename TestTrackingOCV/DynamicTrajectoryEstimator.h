#pragma once
#include "CameraPoseProvider.h"
#include "TrajectoryArchiver.h"
#include "Tracker.h"
#include "Track.h"

class DynamicTrajectoryEstimator
{
public:
  DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider);

  void buildTracks(int frameId, cv::Mat const& img, cv::Mat& outputFrame);
  void loadOnlyDynamicsTracksFromFile(std::string &pathToAllTracks);

  void updateEstimates(int frameId);
  void finilizeTrack(Track& t);
  void registerTrack(Track& t);

protected:
  std::vector<std::shared_ptr<Track>> dynamicTracks;
  std::vector<cv::Point3d> objectPoints;
  std::vector<cv::Point2d> imagePoints;

  std::vector<cv::Point2d>  unPointsF, unPointsL;

  std::vector<cv::Mat> owTs;
  std::vector<std::vector<cv::Mat>> oXs;

  CameraPoseProvider & poseProvider;
  cv::Mat const& K;
  cv::Mat const& dist;

  //std::vector<cv::Point2d> unPointsF, unPointsL;

  void setObjectWorldCoordsOnFrame(cv::Mat const& R, cv::Mat const& t, int frameId);
};

