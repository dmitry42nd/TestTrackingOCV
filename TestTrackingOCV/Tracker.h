#pragma once
#include "opencv2/features2d/features2d.hpp"
#include "Track.h"

class Tracker
{
public:
	Tracker();
	~Tracker();

	void trackWithOrb(cv::Mat& nextImg, cv::Mat& outputFrame, int frameInd);
	void createNewTrack(cv::Point2f point, int frameCnt, cv::KeyPoint keyPt, cv::Mat &desc);
	void saveAllTracks(std::string& pathToSaveFolder);
	void trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg);	
	cv::Mat calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<Track*> &curTracks);
	void detectPoints(int indX, int indY, cv::Mat &m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd);
	//cv::Mat calcGridPointDistribution();

	std::vector<cv::KeyPoint> m_prevKeypoints;
	std::vector<cv::KeyPoint> m_nextKeypoints;

	cv::Mat                   m_prevDescriptors;
	cv::Mat                   m_nextDescriptors;

	cv::BFMatcher *m_orbMatcher;
	cv::ORB *orb;

	std::vector<Track*> prevPoints, curPoints;

	std::vector<Track*> lostTracks;	

	int kltPtThr = 200;
	cv::Mat prevImg;
	int wx, wy;
	std::vector<std::vector<cv::Mat>> detMasks;
	cv::Size imSize;

	cv::FastFeatureDetector *fastDetector;

	double dfx = 567.6;
	double dfy = 570.2;
	double dcx = 324.7;
	double dcy = 250.1;

	double cfx = 535.4;
	double cfy = 539.2;
	double ccx = 320.1;
	double ccy = 247.6;

	double trackThr = 70;

};

