// TestTrackingOCV.cpp : Defines the entry point for the console application.

#include "stdafx.h"

#include <regex>
#include <time.h>

#include "Tracker.h"
#include "CameraPoseProviderTXT.h"
#include "TrajectoryArchiver.h"
#include "DynamicTrajectoryEstimator.h"
#include "TestProgram.h"

#define ID_SHIFT 203
//82 duo1
//57 duo2
//283
//203 uno2

typedef boost::filesystem::path ImgPath;

static const std::regex e("^[0-9]+/.[0-9]+/.(bmp|png)$");

void extractNumbers(std::string fOnly, int &prefInt, int& sufInt)
{
  auto sepInd = fOnly.find(".");
  std::string pref = fOnly.substr(0, sepInd);
  std::string suf = fOnly.substr(sepInd + 1, fOnly.size() - sepInd - 1);
  prefInt = std::stoi(pref);
  sufInt = std::stoi(suf);
}


void getDepthImg(cv::Mat &depthImg, std::vector<ImgPath> const &depthImgsPaths, ImgPath const &rgbImgPath, int imgId)
{
  std::string fName = rgbImgPath.filename().string();
  int prefInt, sufInt;
  int minInd = -1; //depth img index
  long double minPref = 1e100;

  if (depthImgsPaths.size() > 0)
  {
    //TODO: usual 000.png case
    //000.000.png
    if (std::regex_match(fName.c_str(), e))
    {
      extractNumbers(fName, prefInt, sufInt);
      for (decltype(depthImgsPaths.size()) cInd = 0; cInd < depthImgsPaths.size(); cInd++)
      {
        int dPrefInt, dSufInt;
        extractNumbers(depthImgsPaths[cInd].filename().string(), dPrefInt, dSufInt);
        long double val = abs(dPrefInt - prefInt)*1e8 + abs(dSufInt - sufInt);
        if (val < minPref)
        {
          minInd = cInd;
          minPref = val;
        }
      }
      depthImg = cv::imread(depthImgsPaths[minInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
    } else {
      depthImg = cv::imread(depthImgsPaths[imgId].string(), CV_LOAD_IMAGE_ANYDEPTH);
      //std::cout  << "depth map type: " << depthImg.type() << std::endl;
    }
  }
}

int main()
{
#if 0
  cv::Mat depth = cv::imread("000005.png", CV_LOAD_IMAGE_ANYDEPTH);
  std::cout << depth.type() << std::endl;
#else
  cv::FileStorage fs("settings.yaml", cv::FileStorage::READ);

  std::string rootFld            = fs["root"];
  std::string inFld              = fs["inFld"];
  std::string outFld             = fs["outFld"];
  std::string depthFld           = fs["depthFld"];
  std::string trackTypesInfoFld  = fs["trackTypesInfoFld"];
  //std::string lostTracksFld      = fs["lostTracksFld"];
  std::string finalTrackTypesFld = fs["finalTrackTypesFld"];

  std::string pathToCameraPoses = fs["pathToCameraPoses"];
  std::string pathToSavedTracks = fs["pathToSavedTracks"];

  boost::filesystem::current_path(rootFld);

  boost::filesystem::path pathToDepthFld(depthFld);
  boost::filesystem::path pathToInFld(inFld);

  std::vector<ImgPath> depthImgsPaths, rgbImgsPaths;
  copy(boost::filesystem::directory_iterator(pathToDepthFld), boost::filesystem::directory_iterator(), std::back_inserter(depthImgsPaths));
  copy(boost::filesystem::directory_iterator(pathToInFld), boost::filesystem::directory_iterator(), std::back_inserter(rgbImgsPaths));
  sort(rgbImgsPaths.begin(), rgbImgsPaths.end());
  sort(depthImgsPaths.begin(), depthImgsPaths.end());

  CameraPoseProviderTXT poseProvider(pathToCameraPoses);

#if 1
  cv::Size imgSize = cv::imread(rgbImgsPaths.front().string()).size();

  TrajectoryArchiver trajArchiver(pathToSavedTracks);
  Tracker tracker(trajArchiver, poseProvider, imgSize);

  clock_t tStart = clock();

  for (decltype(rgbImgsPaths.size())  imgId = 0; imgId < rgbImgsPaths.size(); imgId++)
  {
    cv::Mat depthImg = cv::Mat::zeros(imgSize, CV_16S);
    getDepthImg(depthImg, depthImgsPaths, rgbImgsPaths[imgId], imgId);

    std::string rgbImgName = rgbImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      tracker.trackWithKLT(ID_SHIFT + imgId, img, outImg, depthImg);

      std::string outImgName = outFld + std::to_string(ID_SHIFT + imgId) + ".png";
      cv::imwrite(outImgName, outImg);
    }
    std::cerr << ID_SHIFT + imgId << std::endl;
  }

  double totalTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  fprintf(stderr, "Total time taken: %.2fs\n", totalTime);
  fprintf(stderr, "Average time per frame taken: %.4fs\n", totalTime / rgbImgsPaths.size());
  fprintf(stderr, "Average fps: %.2fs\n", rgbImgsPaths.size() / totalTime);

  std::ofstream trackOut1("../tracktypes/tt-mean3.txt");
  tracker.generateRocDataMean3(trackOut1, 40);
  std::ofstream trackOut2("../tracktypes/tt-mean2.txt");
  tracker.generateRocDataMean2(trackOut2, 40);
  std::ofstream trackOut3("../tracktypes/tt-max.txt");
  tracker.generateRocDataMax(trackOut3, 40);

  std::cerr << "final tracks types\n";
  for (decltype(rgbImgsPaths.size()) imgId = 0; imgId < rgbImgsPaths.size(); imgId++)
  {
    std::string rgbImgName = rgbImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      tracker.drawFinalPointsTypes(ID_SHIFT + imgId, img, outImg);

      std::string outFTTImgName = finalTrackTypesFld + std::to_string(ID_SHIFT + imgId) + ".png";
      cv::imwrite(outFTTImgName, outImg);
    }
    std::cerr << ID_SHIFT + imgId << std::endl;
  }
#endif

#if 1
  std::cerr << "build dynamic tracks\n";

  std::vector<ImgPath> fImgsPaths;
  copy(boost::filesystem::directory_iterator(finalTrackTypesFld), boost::filesystem::directory_iterator(), std::back_inserter(fImgsPaths));
  sort(fImgsPaths.begin(), fImgsPaths.end());

  DynamicTrajectoryEstimator DTE(poseProvider);
  DTE.loadOnlyDynamicsTracksFromFile(pathToSavedTracks);
  //DTE.buildTrack(208, 248, false);
  DTE.buildTrack(255, 295, false);
  //DTE.buildTrack(300, 335);

  //DTE.buildTrack(90, 130, false);
  //DTE.buildTrack(90, 130, true);
  //DTE.buildTrack(60, 90, false);
  //DTE.buildTrack(60, 90, true);

  //DTE.buildTrack(290, 335);
  //DTE.buildTrack(355, 425);
  //DTE.buildTrack(440, 500);
  //kinect601
  //DTE.buildTrack(605, 639);
  //DTE.buildTrack(660, 702);
  //DTE.buildTrack(725, 765);
  //DTE.buildTrack(788, 826);
  //DTE.buildTrack(863, 895);

  /*for (int imgId = 0; ID_SHIFT + imgId < 654; imgId++)
  {
    std::string rgbImgName = fImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      std::string outFTTImgName = finalTrackTypesFld + std::to_string(ID_SHIFT + imgId) + ".bmp";
      cv::imwrite(outFTTImgName, outImg);
    }
    std::cout << ID_SHIFT + imgId << std::endl;
  }*/
#endif


#endif

  return 0;
}
