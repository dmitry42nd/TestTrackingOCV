#pragma once
#include "CameraPoseProvider.h"
#include <string>
#include <unordered_map>

class CameraPoseProviderTXT :
	public CameraPoseProvider
{
public:
	CameraPoseProviderTXT(std::string& pathToCameraPoses);
	~CameraPoseProviderTXT();

	void getCurrentPose(CameraPose& cameraPose) override;
	void getPoseForFrame(CameraPose& cameraPose, int frameNum) override;

	void setCurrentFrameNumber(int frameNum);


protected:
  void readCameraPosesFromFile(std::string& pathToCameraPoses);
	int frameNum;
};

