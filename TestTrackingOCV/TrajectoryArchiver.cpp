#include "stdafx.h"

#include "TrajectoryArchiver.h"


TrajectoryArchiver::TrajectoryArchiver(std::string &pathToAllTracks) :
	allTracksData(pathToAllTracks)
{}

TrajectoryArchiver::~TrajectoryArchiver()
{
	allTracksData.close();
}


void TrajectoryArchiver::archiveTrajectorySimple(std::shared_ptr<Track> track)
{
	allTracksData << track->type << " " << track->history.size() << " ";
	for (auto p : track->history)
	{
		allTracksData << p->frameId << " " << p->loc.x << " " << p->loc.y << " " << p->depth << " ";
	}
}

#if 0
void TrajectoryArchiver::archiveTrajectory(std::shared_ptr<Track>& track)
{
//check if we have at least N depth values
	int N = 2;
	int depthCnt = 0;
	for (int i = 0; i < track->history.size(); i++)
	{
		CameraPose cameraPose;
		int frmId = track->history[i]->frameId;
		poseProvider.getCameraPoseForFrame(cameraPose, frmId);
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

void TrajectoryArchiver::writeTrajectory(std::shared_ptr<Track>& track, int idNum)
{
	std::string trackFName = pathToStorage + std::to_string(idNum) + ".txt";
	std::ofstream trackOut(trackFName);
	for (int i = 0; i < track->history.size(); i++)
	{
		std::shared_ptr<TrackedPoint> tp = track->history[i];
		CameraPose cameraPose;
		poseProvider.getCameraPoseForFrame(cameraPose, tp->frameId);
		if (tp->depth > 0 && cameraPose.R.cols > 0)
		{
			trackOut << tp->frameId << " ";
			trackOut << tp->loc.x << " ";
			trackOut << tp->loc.y << " ";
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
#endif
