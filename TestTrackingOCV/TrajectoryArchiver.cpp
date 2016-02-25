#include "stdafx.h"

#include "TrajectoryArchiver.h"
#include <opencv2\calib3d\calib3d.hpp>


TrajectoryArchiver::TrajectoryArchiver(CameraPoseProvider &poseProvider, std::string &pathToStorage) : poseProvider(poseProvider), pathToStorage(pathToStorage)
{
	archCnt = 0;
}


TrajectoryArchiver::~TrajectoryArchiver()
{
}

void TrajectoryArchiver::writeTrajectory(std::shared_ptr<Track>& track, int idNum)
{
	std::string trackFName = pathToStorage + std::to_string(idNum) + ".txt";
	std::ofstream trackOut(trackFName);
	for (int i = 0; i < track->history.size(); i++)
	{
		std::shared_ptr<TrackedPoint> tp = track->history[i];
		CameraPose cameraPose;
		poseProvider.getPoseForFrame(cameraPose, tp->frameId);
		if (tp->depth > 0 && cameraPose.R.cols > 0)
		{
			trackOut << tp->frameId << " ";
			trackOut << tp->location.x << " ";
			trackOut << tp->location.y << " ";
			trackOut << tp->depth << " ";
			cv::Mat rodVect;
			cv::Rodrigues(cameraPose.R, rodVect);
			for (int i = 0; i < 3; i++)
			{
				trackOut << rodVect.at <double>(i, 0) << " ";
			}
			for (int i = 0; i < 3; i++)
			{
				trackOut << cameraPose.t.at <double>(i, 0) << " ";
			}
		}
	}
}

void TrajectoryArchiver::writeTrajectorySimple(std::shared_ptr<Track>& track, int idNum)
{
  std::string trackFName = pathToStorage + std::to_string(idNum) + ".txt";
  std::cout << trackFName << std::endl;
  std::ofstream trackOut(trackFName);
  for (int i = 0; i < track->history.size(); i++)
  {
    std::shared_ptr<TrackedPoint> tp = track->history[i];
    trackOut << tp->frameId << " ";
    trackOut << tp->location.x << " ";
    trackOut << tp->location.y << " ";
    trackOut << tp->depth << " ";
  }
}

void TrajectoryArchiver::archiveTrajectorySimple(std::shared_ptr<Track>& track)
{
  writeTrajectorySimple(track, archCnt);
  archCnt++;
}

void TrajectoryArchiver::archiveTrajectory(std::shared_ptr<Track>& track)
{
//check if we have at least N depth values
	int N = 2;
	int depthCnt = 0;
	for (int i = 0; i < track->history.size(); i++)
	{
		CameraPose cameraPose;
		int frmId = track->history[i]->frameId;
		poseProvider.getPoseForFrame(cameraPose, frmId);
		if (track->history[i]->depth > 0)
		{
			//std::cout << " non-zero depth !" << std::endl;
		}
		if (cameraPose.R.cols > 0)
		{
//			std::cout << " non-zero pose !" << std::endl;
		}
		if (track->history[i]->depth>0 && cameraPose.R.cols > 0)
		{
			depthCnt++;
		}
	}
	if (depthCnt >= N)
	{
		writeTrajectory(track, archCnt);
		archCnt++;
	}
}
