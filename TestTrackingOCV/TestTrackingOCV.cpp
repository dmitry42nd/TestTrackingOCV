// TestTrackingOCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "Tracker.h"

#include "boost/filesystem.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

void saveAllDepth()
{
	std::string depthFld = "C:\\projects\\kkdata\\depth\\";
	std::string depthDebFld = "C:\\projects\\kkdata\\depthDebugAll\\";
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

void extractNumbers(std::string fOnly, int &prefInt, int& sufInt)
{	
	int sepInd = fOnly.find(".");
	std::string pref = fOnly.substr(0, sepInd);
	std::string suf = fOnly.substr(sepInd + 1, fOnly.size() - sepInd - 1);
	prefInt = std::stoi(pref);
	sufInt = std::stoi(suf);
}


void saveDepthData()
{

	std::string dirName = "C:\\projects\\kkdata\\fullTrack\\rgb\\";
	std::string outDirName = "C:\\projects\\kkdata\\out\\";
	std::string depthFld = "C:\\projects\\kkdata\\depth\\";
	std::string depthDebFld = "C:\\projects\\kkdata\\depthDebug\\";
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

	std::string folder = "C:\\projects\\kkdata\\tracks_6_11\\key_tracks\\";
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
					depthVal = (int)depthImg.at<ushort>(round(pt.y), round(pt.x));
					depthVal /= 5000.0;
				}
			}
			std::cout << depthVal << std::endl;
			fOut << depthVal << std::endl;
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	//saveDepthData();
	//return 0;

	//saveAllDepth();

	std::string dirName = "C:\\projects\\kkdata\\fullTrack\\rgb\\";
	std::string outDirName = "c:\\projects\\debug_tracking\\out\\";
	std::string outCleanDirName = "C:\\projects\\kkdata\\outClean\\";
	std::string depthFld = "C:\\projects\\kkdata\\depth\\";
	std::string depthDebFld = "C:\\projects\\kkdata\\depthDebug\\";
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
	Tracker tracker;
	while (dInd < vRgb.size())
	{
		cv::Mat depthImg;
		
		std::string fName = vRgb[dInd].string();

		std::string fNameOnly = vRgb[dInd].filename().string();
		int prefInt, sufInt;
		extractNumbers(fNameOnly, prefInt, sufInt);
		
		int minInd=-1;
		double minPref=1e100;
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

		std::cout << fName << std::endl;
		if (boost::filesystem::exists(fName))
		{
			cv::Mat img = cv::imread(fName, 0);			
			cv::Mat outputImg;
			cv::cvtColor(img, outputImg, CV_GRAY2BGR);

			depthImg = cv::imread(v[minInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
			//std::cout << depthImg.channels() << std::endl;
			//std::cout << depthImg.type() << " " << CV_16U << std::endl;
			//std::cout << depthImg.at<short>(480-1, 640-1) << std::endl;
			cv::imwrite(depthDebFld + "f" + std::to_string(dInd) + ".bmp", 255/10*depthImg/5000);

			std::string fNameOutClean = outCleanDirName + std::to_string(dInd) + ".bmp";
			cv::imwrite(fNameOutClean, outputImg);

			tracker.trackWithKLT(img, outputImg, dInd, depthImg);
			std::string fNameOut = outDirName + std::to_string(dInd) + ".bmp";
			cv::imwrite(fNameOut, outputImg);
		}
		std::cout << dInd << " " << tracker.lostTracks.size() << std::endl;
		dInd++;
	}
	std::string pathToSave = "C:\\projects\\kkdata\\trackLogFull\\";
	tracker.saveAllTracks(pathToSave);
	return 0;
}
