#pragma once
#include "Track.h"
#include "TrajectoryArchiver.h"
class Tracker
{
public:
	Tracker(TrajectoryArchiver &trajArchiver, cv::Size imSize);

  void trackWithOrb(cv::Mat & m_nextImg, cv::Mat & outputFrame, int frameInd, cv::Mat & depthImg);
  void detectPointsOrb(int indX, int indY, cv::Mat & m_nextImg, cv::Mat & depthImg, cv::Mat & outputFrame, int frameInd);
	void createNewTrack(cv::Point2f point, int frameCnt, cv::KeyPoint const &keyPt, cv::Mat const &desc, double depth = 0);
	void saveAllTracks(std::string& pathToSaveFolder);
	void trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg);	
	cv::Mat calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks);
	void detectPoints(int indX, int indY, cv::Mat &m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd);
  void defineTrackType(std::shared_ptr<Track> & track);
	//cv::Mat calcGridPointDistribution();

	std::vector<cv::KeyPoint> m_prevKeypoints;
	std::vector<cv::KeyPoint> m_nextKeypoints;

  cv::Mat                   m_tracksFrame;
	cv::Mat                   m_prevDescriptors;
	cv::Mat                   m_nextDescriptors;

	cv::BFMatcher *m_orbMatcher;
	cv::ORB *orb;

	std::vector<std::shared_ptr<Track>> prevTracks, curTracks;

	std::vector<std::shared_ptr<Track>> lostTracks;

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

	double trackThr = 30;

protected:
  typedef std::pair<int, int> Coords;

	std::vector<cv::KeyPoint> filterPoints(int wx, int wy, std::vector<cv::KeyPoint>& keyPts);
  cv::Mat curFrameProjMatr;
  std::vector<cv::KeyPoint> curKeyPts;

private:
	TrajectoryArchiver trajArchiver;

  int mcnt;
};

