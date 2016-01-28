#pragma once
#include "CameraPoseProvider.h"
#include <string>
#include <unordered_map>

class CameraPoseProviderTXT :
	public CameraPoseProvider
{
public:
	CameraPoseProviderTXT(std::string& pathToTracksFolder);
	~CameraPoseProviderTXT();

	void getCurrentPose(CameraPose& cameraPose) override;
	void getPoseForFrame(CameraPose& cameraPose, int frameNum) override;

	void setCurrentFrameNumber(int frameNum);
private:
	std::unordered_map<int, cv::Mat> poses;
	int frameNum;
};

