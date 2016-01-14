#pragma once
#include "CameraPoseProvider.h"
#include "TrajectoryArchiver.h"

class DynamicTrajectoryEstimator
{
public:
  DynamicTrajectoryEstimator();
  DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider, TrajectoryArchiver& arch);
  
};

