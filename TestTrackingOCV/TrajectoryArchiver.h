#pragma once
#include "CameraPoseProvider.h"
#include "Track.h"

class TrajectoryArchiver
{
public:  
  TrajectoryArchiver(CameraPoseProvider &poseProvider, std::string &pathToStorage);

  void archiveTrajectory(std::shared_ptr<Track>& track);
  ~TrajectoryArchiver();

private:
	void writeTrajectory(std::shared_ptr<Track>& track, int idNum);

	CameraPoseProvider &poseProvider;
	std::string pathToStorage;
	int archCnt;
};

