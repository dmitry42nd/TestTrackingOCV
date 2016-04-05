#pragma once
#include "CameraPoseProvider.h"
#include "Track.h"

class TrajectoryArchiver
{
public:  
  TrajectoryArchiver(std::string &pathToSavedTracks);
	~TrajectoryArchiver();

  void archiveTrajectorySimple(std::shared_ptr<Track> track);
	//void archiveTrajectory(std::shared_ptr<Track>& track);

private:
	std::ofstream allTracksData;
	//void writeTrajectory(std::shared_ptr<Track>& track, int idNum);
};

