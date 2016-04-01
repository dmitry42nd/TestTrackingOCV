#pragma once
#include "Track.h"
#include "TrajectoryArchiver.h"

class Tracker
{
public:
	Tracker(TrajectoryArchiver &trajArchiver, cv::Size imSize);
  Tracker(TrajectoryArchiver &trajArchiver, cv::Size imSize, std::string pathToTrackTypes);

	void trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg);
  void trackWithOrb(cv::Mat & m_nextImg, cv::Mat & outputFrame, int frameInd, cv::Mat & depthImg);
  void detectPointsOrb(int indX, int indY, cv::Mat & m_nextImg, cv::Mat & depthImg, cv::Mat & outputFrame, int frameInd);
	void generateRocData(std::ofstream &file, int maxThrErr);
	void saveAllTracks(std::string& pathToSaveFolder);
	void drawFinalPointsTypes(cv::Mat &m_nextImg, cv::Mat &outputFrame, int frameInd, cv::Mat &depthImg);

	void loadAllTracks(std::string &pathToAllTracks);
	void buildTracks(cv::Mat &m_nextImg, cv::Mat &outputFrame, int frameInd);

	//cv::Mat calcGridPointDistribution();

	std::vector<cv::KeyPoint> m_prevKeypoints;
	std::vector<cv::KeyPoint> m_nextKeypoints;

  cv::Mat                   m_tracksFrame;
	cv::Mat                   m_prevDescriptors;
	cv::Mat                   m_nextDescriptors;

	cv::BFMatcher *m_orbMatcher;
	cv::ORB *orb;

	std::vector<std::shared_ptr<Track>> prevTracks, curTracks, lostTracks;

	int kltPtThr = 200;
	cv::Mat prevImg;
	int wx, wy;
	std::vector<std::vector<cv::Mat>> detMasks;
	cv::Size imSize;

	double dfx = 567.6;
	double dfy = 570.2;
	double dcx = 324.7;
	double dcy = 250.1;

	double cfx = 535.4;
	double cfy = 539.2;
	double ccx = 320.1;
	double ccy = 247.6;

	double trackThr = 70;

protected:
  typedef std::pair<int, int> Coords;

	void createNewTrack(cv::Point2f point, int frameCnt, cv::KeyPoint const &keyPt, cv::Mat const &desc, double depth = 0);
	cv::Mat calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks);
	void detectPoints(int indX, int indY, cv::Mat &m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd);
	std::vector<cv::KeyPoint> filterPoints(int wx, int wy, std::vector<cv::KeyPoint>& keyPts);
	void defineTrackType(std::shared_ptr<Track> track, double errThr);

  std::string pathToTrackTypes;
	std::vector<std::pair<double,bool>> errs_v;
	cv::Ptr<cv::FastFeatureDetector> fastDetector;

private:
	TrajectoryArchiver trajArchiver;
};

