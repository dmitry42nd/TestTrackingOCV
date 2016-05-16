#pragma once
#include "Track.h"
#include "TrajectoryArchiver.h"

class Tracker
{
public:
	//Tracker(TrajectoryArchiver const& trajArchiver, CameraPoseProvider const& poseProvider, cv::Size imSize);
	Tracker(TrajectoryArchiver & trajArchiver, CameraPoseProvider & poseProvider, cv::Size imSize);

	void trackWithKLT(int frameId, cv::Mat const& img, cv::Mat& outImg, cv::Mat& depthImg);
	void drawFinalPointsTypes(int frameId, cv::Mat const& img, cv::Mat &outImg);

	void generateRocDataMean3(std::ofstream &file, int maxThrErr);
	void generateRocDataMean2(std::ofstream &file, int maxThrErr);
	void generateRocDataMax(std::ofstream &file, int maxThrErr);

	//orb stuff
#if 0
	void trackWithOrb(cv::Mat & m_nextImg, cv::Mat & outputFrame, int frameInd, cv::Mat & depthImg);
	std::vector<cv::KeyPoint> m_prevKeypoints;
	std::vector<cv::KeyPoint> m_nextKeypoints;

  cv::Mat                   m_tracksFrame;
	cv::Mat                   m_prevDescriptors;
	cv::Mat                   m_nextDescriptors;

	cv::BFMatcher *m_orbMatcher;
	cv::ORB *orb;
#endif

protected:
	std::ofstream errMean3File;

	void generateRocData(std::ofstream &file, int maxThrErr, std::vector<std::pair<double,bool>> const & errs);

	typedef std::pair<int, int> Coords;

	std::vector<std::shared_ptr<Track>> prevTracks, curTracks;
	std::vector<std::shared_ptr<Track>> lostTracks;

	cv::Mat prevImg;
	cv::Size imgSize;

	const int kltPointsMin   = 200;
	// for dte uno2
	/*const double optFlowThr  = 40;
	const double backProjThr = 5;
	const double KLTErrThr	 = 0.2;*/

	const double optFlowThr  = 40;
	const double backProjThr = 25;
	const double KLTErrThr	 = 0.2;

  const int wx = 2;
	const int wy = 2;

	std::vector<std::pair<double,bool>> errs_mean2;
	std::vector<std::pair<double,bool>> errs_mean3;
	std::vector<std::pair<double,bool>> errs_max;

	cv::Ptr<cv::FastFeatureDetector> fastDetector;
	cv::Mat const& K;
	cv::Mat const& dist;


	std::vector<std::vector<cv::Mat>> detMasks;

	void createNewTrack(cv::Point2f point, int frameId, cv::KeyPoint const &keyPt, cv::Mat const &desc, double depth = 0);
	cv::Mat calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks);
	void detectPoints(int indX, int indY, cv::Mat const& img, cv::Mat& depthImg, cv::Mat& outImg, int frameId);
	std::vector<cv::KeyPoint> filterPoints(int wx, int wy, std::vector<cv::KeyPoint>& keyPts);
	void defineTrackType(std::shared_ptr<Track> track);
	void undistPoint(cv::Point2f const& point, cv::Point2d & undist);
	void getProjectionAndNormCeres(double *camera, double *point, cv::Point2f &pp, cv::Point3f &np);
private:
	TrajectoryArchiver & trajArchiver;
	CameraPoseProvider & poseProvider;
};