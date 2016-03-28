// TestTrackingOCV.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <regex>

#include <time.h>
#include "Tracker.h"

#include "CameraPoseProviderTXT.h"
#include "TrajectoryArchiver.h"

#if 0
void saveAllDepth()
{
  std::string depthFld = "../../depth/";
  std::string depthDebFld = "../../depthDebugAll/";
  boost::filesystem::path p(depthFld);
  typedef std::vector<boost::filesystem::path> vec;             // store paths,
  vec v;                                // so we can sort them later
  copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), std::back_inserter(v));
  sort(v.begin(), v.end());
  int dInd = 0;
  while (!boost::filesystem::is_regular_file(v[dInd]))
  {
    dInd++;
  }
  std::cout << v[dInd] << std::endl;
  for (int i = 0; i < 3000; i++)
  {
    if (dInd >= v.size())
    {
      break;
    }
    cv::Mat depthImg = cv::imread(v[dInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
    cv::imwrite(depthDebFld + "f" + std::to_string(i) + ".bmp", 255 / 10 * depthImg / 5000);
    dInd++;
  }
}
#endif

void extractNumbers(std::string fOnly, int &prefInt, int& sufInt)
{
  auto sepInd = fOnly.find(".");
  std::string pref = fOnly.substr(0, sepInd);
  std::string suf = fOnly.substr(sepInd + 1, fOnly.size() - sepInd - 1);
  prefInt = std::stoi(pref);
  sufInt = std::stoi(suf);
}

#if 0
void saveDepthData()
{

  std::string dirName = "../../fullTrack/rgb/";
  std::string depthFld = "../../depth/";
  std::string outDirName = "../../out/";
  std::string depthDebFld = "../../depthDebug/";

  boost::filesystem::path p(depthFld);
  boost::filesystem::path p2(dirName);

  typedef std::vector<boost::filesystem::path> vec;             // store paths,
  vec v, vRgb;                                // so we can sort them later
  copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), std::back_inserter(v));
  copy(boost::filesystem::directory_iterator(p2), boost::filesystem::directory_iterator(), std::back_inserter(vRgb));
  sort(v.begin(), v.end());
  sort(vRgb.begin(), vRgb.end());

  int dInd = 0;
  while (!boost::filesystem::is_regular_file(vRgb[dInd]))
  {
    dInd++;
  }
  std::cout << v[dInd] << std::endl;
  std::vector<int> depthInds;
  while (dInd < vRgb.size())
  {
    cv::Mat depthImg;

    std::string fName = vRgb[dInd].string();

    std::string fNameOnly = vRgb[dInd].filename().string();
    int prefInt, sufInt;
    extractNumbers(fNameOnly, prefInt, sufInt);

    int minInd = -1;
    double minPref = 1e100;
    for (int cInd = 0; cInd < v.size(); cInd++)
    {
      int dPrefInt, dSufInt;
      extractNumbers(v[cInd].filename().string(), dPrefInt, dSufInt);
      long double val = abs(dPrefInt - prefInt)*1e8 + abs(dSufInt - sufInt);
      if (val < minPref)
      {
        minInd = cInd;
        minPref = val;
      }
    }
    depthInds.push_back(minInd);
    dInd++;
  }

  std::string folder = "../../tracks_6_11/key_tracks/";
  for (int i = 0; i < 1447; i++)
  {
    std::string fName = folder + "t" + std::to_string(i) + ".txt";
    std::string fNameOut = folder + "d" + std::to_string(i) + ".txt";
    std::ifstream fIn(fName);
    std::ofstream fOut(fNameOut);
    double coord;
    fIn >> coord;
    fIn >> coord;
    fIn >> coord;
    while (!fIn.eof())
    {
      int frameId;
      fIn >> frameId;
      if (fIn.eof())
      {
        break;
      }
      cv::Point2d pt;
      fIn >> pt.x;
      fIn >> pt.y;
      double depthVal = 0;
      if (frameId < 1067)
      {
        int dInd = depthInds[frameId];
        std::cout << dInd << std::endl;
        cv::Mat depthImg = cv::imread(v[dInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
        if (pt.x > 0 && pt.y > 0 && pt.x < depthImg.cols && pt.y < depthImg.rows)
        {
          depthVal = (int)depthImg.at<ushort>(floor(pt.y), floor(pt.x));
          depthVal /= 5000.0;
        }
      }
      std::cout << depthVal << std::endl;
      fOut << depthVal << std::endl;
    }
  }
}
#endif

static const std::regex e("^[0-9]+/.[0-9]+/.(bmp|png)$");
typedef std::vector<boost::filesystem::path> vec; // store paths, so we can sort them later

#define PI 3.14159265

int main()
{
#if 0
  double data[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
  cv::Mat pose = cv::Mat(3,4,CV_64F, data);
  std::cout << pose << "\n";
  CameraPose cameraPose;
  cameraPose.R = pose(cv::Rect(0, 0, 3, 3));
  cameraPose.t = pose(cv::Rect(3, 0, 1, 3));
  std::cout << cameraPose.R << "\n";
  std::cout << cameraPose.t << "\n";
  
#else
  clock_t tStart = clock();

  boost::filesystem::current_path("/home/dmitry/projects/DynTrack/TestTrackingOCV/bin");

  std::string dirName = "../../fullTrack/rgb/";
  std::string outDirName = "../../debug_tracking/out/";
  std::string outCleanDirName = "../../outClean/";
  std::string depthFld = "../../depth/";
  std::string depthDebFld = "../../depthDebug/";
  std::string pathToTracksFolder = "../../tracks_6_11/track/";
  std::string pathToStorage = "../../TD_Data/";
  std::string pathToPostProc = "../../outProc/";

  
  std::string pathToCameraPoses = "../../cameraPoses";
  std::string pathToTrackTypes = "../../tracktypes/";

  CameraPoseProviderTXT poseProvider(pathToCameraPoses);
  TrajectoryArchiver trajArchiver(poseProvider, pathToStorage);

  boost::filesystem::path p(depthFld);
  boost::filesystem::path p2(dirName);

  vec v, vRgb;
  copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), std::back_inserter(v));
  copy(boost::filesystem::directory_iterator(p2), boost::filesystem::directory_iterator(), std::back_inserter(vRgb));
  sort(v.begin(), v.end());
  sort(vRgb.begin(), vRgb.end());

  int dInd = 0;
  while (!boost::filesystem::is_regular_file(vRgb[dInd]))
  {
    dInd++;
  }

  cv::Size imgsSize = cv::imread(vRgb[dInd].string()).size();
  Tracker tracker(trajArchiver, imgsSize, pathToTrackTypes);

  while (dInd < vRgb.size())
  {
    poseProvider.setCurrentFrameNumber(dInd);

    cv::Mat depthImg;
    std::string fName = vRgb[dInd].string();
    std::string fNameOnly = vRgb[dInd].filename().string();
    int prefInt, sufInt;
    int minInd = -1; //depth img index
    long double minPref = 1e100;

    depthImg = cv::Mat::zeros(imgsSize, CV_16S);

    if (v.size() > 0) {
      //TODO: usual 000.png case
      //000.000.png
      if (std::regex_match(fNameOnly.c_str(), e)) {
        extractNumbers(fNameOnly, prefInt, sufInt);
        for (int cInd = 0; cInd < v.size(); cInd++)
        {
          int dPrefInt, dSufInt;
          extractNumbers(v[cInd].filename().string(), dPrefInt, dSufInt);
          long double val = abs(dPrefInt - prefInt)*1e8 + abs(dSufInt - sufInt);
          if (val < minPref)
          {
            minInd = cInd;
            minPref = val;
          }
        }

        depthImg = cv::imread(v[minInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
        //std::cout << depthImg.channels() << std::endl;
        //std::cout << depthImg.type() << " " << CV_16U << std::endl;
        //std::cout << depthImg.at<short>(480-1, 640-1) << std::endl;
        cv::imwrite(depthDebFld + "f" + std::to_string(dInd) + ".bmp", 255 / 10 * depthImg / 5000);
      }
    }

#define ID_SHIFT 601
    std::cout << fName << std::endl;
    if (boost::filesystem::exists(fName))
    {
      cv::Mat img = cv::imread(fName, 0);
      cv::Mat outputImg;
      cv::cvtColor(img, outputImg, CV_GRAY2BGR);

      std::string fNameOutClean = outCleanDirName + std::to_string(dInd) + ".bmp";
      cv::imwrite(fNameOutClean, outputImg);

      tracker.trackWithKLT(img, outputImg, ID_SHIFT + dInd, depthImg);
      //tracker.trackWithOrb(img, outputImg, dInd, depthImg);
      std::string fNameOut = outDirName + std::to_string(ID_SHIFT + dInd) + ".bmp";
      cv::imwrite(fNameOut, outputImg);
    }
    std::cout << ID_SHIFT + dInd << " " << tracker.lostTracks.size() << std::endl;
    dInd++;
  }

  //
  std::cerr << "postprocessing stuff\n";
  dInd = 0;
  while (!boost::filesystem::is_regular_file(vRgb[dInd]))
  {
    dInd++;
  }

  while (dInd < vRgb.size())
  {
    poseProvider.setCurrentFrameNumber(dInd);

    cv::Mat depthImg;
    std::string fName = vRgb[dInd].string();
    std::string fNameOnly = vRgb[dInd].filename().string();

    std::cout << fName << std::endl;
    if (boost::filesystem::exists(fName))
    {
      cv::Mat img = cv::imread(fName, 0);
      cv::Mat outputImg;
      cv::cvtColor(img, outputImg, CV_GRAY2BGR);

      tracker.postProcessing(img, outputImg, ID_SHIFT + dInd, depthImg);

      std::string fNameOut = pathToPostProc + std::to_string(ID_SHIFT + dInd) + ".bmp";
      cv::imwrite(fNameOut, outputImg);
    }
    std::cout << ID_SHIFT + dInd << " " << tracker.lostTracks.size() << std::endl;
    dInd++;
  }



  double totalTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;

  std::string pathToSave = "../../trackLogFull/";
  tracker.saveAllTracks(pathToSave);

  std::ofstream trackOut("../../tracktypes/tt-mean3-gen.txt");
  tracker.generateRocData(trackOut, 300);

  fprintf(stderr, "Total time taken: %.2fs\n", totalTime);
  fprintf(stderr, "Average time per frame taken: %.4fs\n", totalTime / vRgb.size());
  fprintf(stderr, "Average fps: %.2fs\n", vRgb.size() / totalTime);
#endif

  std::cout << "Done" << std::endl;

  return 0;
}
