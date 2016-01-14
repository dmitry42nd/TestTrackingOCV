#pragma once
#include "CameraPoseProvider.h"
#include "TrajectoryArchiver.h"
#include "Track.h"

class DynamicTrajectoryEstimator
{
public:
  DynamicTrajectoryEstimator();
  DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider, TrajectoryArchiver& arch);

  void updateEstimates(int frameId);
  void finilizeTrack(Track& t);
  void registerTrack(Track& t);
  
};

