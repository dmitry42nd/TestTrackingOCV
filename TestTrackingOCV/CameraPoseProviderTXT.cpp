#include "stdafx.h"
#include "CameraPoseProviderTXT.h"
#include "boost/filesystem.hpp"
#include <map>
#include <utility>
namespace fs = boost::filesystem;

CameraPoseProviderTXT::CameraPoseProviderTXT(std::string& pathToTracksFolder)
{
	poses.clear();	
	fs::path trackFolder(pathToTracksFolder);
	fs::directory_iterator endIt;	
	for (fs::directory_iterator dirIt(trackFolder); dirIt != endIt; ++dirIt)
	{
		if (fs::is_regular_file(dirIt->status()))
		{
			std::string trackFName(dirIt->path().filename().string());
			std::string trackFullPath(dirIt->path().string());
			std::string trackNum = trackFName.substr(1, trackFName.find(".")-1);
			int trackInd = atoi(trackNum.c_str());
			cv::Mat poseMat = cv::Mat::zeros(4, 4, CV_64F);
			std::ifstream poseIn(trackFullPath);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{					
					poseIn >> poseMat.at<double>(i, j);
				}
			}
			for (int i = 0; i < 3; i++)
			{
				poseIn >> poseMat.at<double>(i, 3);
			}
			poseMat.at<double>(3, 3) = 1.0;
			poses.insert(std::make_pair(trackInd, poseMat));
		}
	}
	frameNum = 0;
}

void CameraPoseProviderTXT::setCurrentFrameNumber(int frameInd)
{
	frameNum = frameInd;
}


CameraPoseProviderTXT::~CameraPoseProviderTXT()
{
}

void CameraPoseProviderTXT::getCurrentPose(CameraPose& cameraPose)
{	
	getPoseForFrame(cameraPose, frameNum);
}

void CameraPoseProviderTXT::getPoseForFrame(CameraPose& cameraPose, int qFrameNum)
{
	cameraPose.R = cv::Mat::zeros(0, 0, CV_64F);
	cameraPose.t = cv::Mat::zeros(0, 0, CV_64F);
	if (poses.count(qFrameNum) > 0)
	{
		std::cout << "pose given " << std::endl;
		cameraPose.R = poses[qFrameNum](cv::Rect(0, 0, 3, 3));
		cameraPose.t = poses[qFrameNum](cv::Rect(3, 0, 1, 3));
	}
}
