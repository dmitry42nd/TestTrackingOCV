#pragma once
#include "CameraPoseProvider.h"
#include <string>
#include <unordered_map>

class CameraPoseProviderTXT :
	public CameraPoseProvider
{
public:
	CameraPoseProviderTXT(std::string& pathToCameraPoses);

  int getCameraPoseForFrame(CameraPose &cameraPose, int frameId) override;
  int getProjMatrForFrame(cv::Mat & projMatr, int frameId) override;
  //int getCeresCameraForFrame(double * camera, int frameNum) override;

protected:
  //void makeCeresCamera(double* camera, CameraPose const& cp);

};

