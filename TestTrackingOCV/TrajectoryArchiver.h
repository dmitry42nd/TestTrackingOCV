#pragma once
#include "CameraPoseProvider.h"
#include "Track.h"

class TrajectoryArchiver
{
public:  
  TrajectoryArchiver(CameraPoseProvider &poseProvider, std::string &pathToStorage);

  void archiveTrajectory(std::shared_ptr<Track>& track);
  void archiveTrajectorySimple(std::shared_ptr<Track>& track);

  ~TrajectoryArchiver();
  CameraPoseProvider &poseProvider;

private:
	void writeTrajectory(std::shared_ptr<Track>& track, int idNum);
  void writeTrajectorySimple(std::shared_ptr<Track>& track, int idNum);

  std::string pathToStorage;
	int archCnt;
};

