#include "Tracker.h"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

Tracker::Tracker()
{
	orb = new cv::ORB(1000, 1.2, 8, 51);
	m_orbMatcher = new cv::BFMatcher(cv::NORM_HAMMING);
	factDetector = new cv::FastFeatureDetector(10);
	wx = 2;
	wy = 2;

	imSize = cv::Size(640, 480);
	for (int j = 0; j < wy; j++)
	{
		detMasks.push_back(std::vector<cv::Mat>());
		for (int i = 0; i < wx; i++)
		{
			detMasks[j].push_back(cv::Mat());
		}
	}	
	int sx = imSize.width / wx;
	int sy = imSize.height/ wy;
	for (int i = 0; i < wx; i++)
	{
		for (int j = 0; j < wy; j++)
		{
			cv::Mat mask = cv::Mat::zeros(imSize, CV_8U);
			for (int px = 0; px < imSize.width; px++)
			{
				for (int py = 0; py < imSize.height; py++)
				{
					if (px / sx == i && py / sy == j)
					{
						mask.at<uchar>(py, px) = 255;
					}
				}
			}
			detMasks[j][i] = mask;
			//cv::imwrite("C:\\projects\\debug_tracking\\mask.bmp", mask);
		}
	}

}


Tracker::~Tracker()
{
}

void Tracker::createNewTrack(cv::Point2f point, int frameCnt, cv::KeyPoint keyPt, cv::Mat &desc)
{
	Track *newTrack = new Track();
	newTrack->history.push_back(new TrackedPoint(point, frameCnt, 0, keyPt, desc));
	newTrack->bestCandidate = newTrack->history[0];
	curPoints.push_back(newTrack);
}

void roundCoords(int &px, int &py, cv::Point2f &pt, cv::Mat &img)
{
	px = round(pt.x);
	px = std::min({ img.cols - 1, px });
	px = std::max({ 0, px });
	py = round(pt.y);
	py = std::min({ img.rows - 1, py });
	py = std::max({ 0, py });
}

cv::Point2f fillDepthPt(cv::Point2f& ptIm, double ccx, double ccy, double cfx, double cfy, double dcx, double dcy, double dfx, double dfy)
{
	cv::Point2f pt;
	pt.x = (ptIm.x - ccx) / cfx;
	pt.y = (ptIm.y - ccy) / cfy;
	pt.x = dfx*pt.x + dcx;
	pt.y = dfy*pt.y + dcy;
	return pt;
}

cv::Mat Tracker::calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<Track*> &curTracks)
{
	cv::Mat res = cv::Mat::zeros(wy, wx, CV_32F);
	int sx = imSize.width / wx;
	int sy = imSize.height/ wy;
	for (int i = 0; i < curTracks.size(); i++)
	{
		int resX = curTracks[i]->bestCandidate->location.x / sx;
		int resY = curTracks[i]->bestCandidate->location.y / sy;
		res.at<float>(resY, resX) = res.at<float>(resY, resX) + 1;
	}
	cv::Mat boolRes = cv::Mat::zeros(wy, wx, CV_8U);
	for (int i = 0; i < wx; i++)
	{
		for (int j = 0; j < wy; j++)
		{
			if (res.at<float>(j, i) < ptNum)
			{
				boolRes.at<uchar>(j, i) = 1;
			}
		}
	}
	return boolRes;
}

void Tracker::detectPoints(int indX, int indY, cv::Mat& m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd)
{
	std::vector<cv::KeyPoint> keyPts;
	std::cout << " detecting.. " << indX << " " << indY << std::endl;
	factDetector->detect(m_nextImg, keyPts, detMasks[indY][indX]);
	for (int i = 0; i < keyPts.size(); i++)
	{
		//std::cout << "init depth read " << std::endl;
		int px, py;

		cv::Point2f pt = fillDepthPt(keyPts[i].pt, ccx, ccy, cfx, cfy, dcx, dcy, dfx, dfy);
		roundCoords(px, py, pt, m_nextImg);
		double depthVal = (int)depthImg.at<ushort>(py, px);
		depthVal /= 5000.0;
		TrackedPoint *tpt = new TrackedPoint(keyPts[i].pt, frameInd, 0, keyPts[i], cv::Mat(), depthVal);
		Track* newTrack = new Track();
		newTrack->history.push_back(tpt);
		newTrack->bestCandidate = tpt;
		curPoints.push_back(newTrack);
		cv::circle(outputFrame, keyPts[i].pt, 3, cv::Scalar(255, 0, 0));
	}

}

void Tracker::trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg)
{	

	curPoints.clear();
	cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);
	if (prevPoints.size() > 0)
	{		
		std::vector<cv::Point2f> prevCorners;
		for (int i = 0; i < prevPoints.size(); i++)
		{
			int hLen = prevPoints[i]->history.size();
			prevCorners.push_back(prevPoints[i]->history[hLen - 1]->location);
		}
		std::vector<cv::Point2f> nextCorners;
		std::vector<uchar> status;
		std::vector<float> err;
		double minEigThreshold = 1e-2;
		cv::calcOpticalFlowPyrLK(prevImg, m_nextImg, prevCorners, nextCorners, status, err,
			cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
			cv::OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold);
		for (int i = 0; i < prevPoints.size(); i++)
		{
			cv::Mat err = cv::Mat(nextCorners[i] - prevCorners[i]);
			double trackDist = norm(err);
			if (status[i] && nextCorners[i].x >= 0 && nextCorners[i].x < m_nextImg.cols &&
				nextCorners[i].y >= 0 && nextCorners[i].y < m_nextImg.rows && trackDist < trackThr)
			{						
				//std::cout << "track depth read " << round(nextCorners[i].y) << " " << round(nextCorners[i].x) << std::endl;
				//std::cout << nextCorners[i] << std::endl;
				cv::Point2f pt = fillDepthPt(nextCorners[i], ccx, ccy, cfx, cfy, ccx, ccy, cfx, cfy);
				int px, py;
				
				//std::cout << pt<< std::endl;
				roundCoords(px, py, pt, m_nextImg);
				double depthVal = 0;
				if (pt.x > 0 && pt.y > 0 && pt.x < depthImg.cols && pt.y < depthImg.rows)
				{
					depthVal = (int)depthImg.at<ushort>(py, px);
					depthVal /= 5000.0;
				}
				prevPoints[i]->bestCandidate = new TrackedPoint(nextCorners[i], frameInd, 0, cv::KeyPoint(), cv::Mat(), depthVal);
				prevPoints[i]->history.push_back(prevPoints[i]->bestCandidate);
				curPoints.push_back(prevPoints[i]);

				cv::circle(outputFrame, prevCorners[i], 5, cv::Scalar(250, 0, 250), -1);
				cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
				cv::circle(outputFrame, nextCorners[i], 3, cv::Scalar(0, 250, 0), -1);
			}
			else 
			{
				if (prevPoints[i]->history.size() > 3)
				{
					lostTracks.push_back(prevPoints[i]);
				}				
			}
		}
	}

	cv::Mat boolStats = calcStatsByQuadrant(wx, wy, kltPtThr / 4, curPoints);

	for (int i = 0; i < wx; i++)
	{
		for (int j = 0; j < wy; j++)
		{
			if (boolStats.at<uchar>(j, i) > 0)
			{
				detectPoints(i, j, m_nextImg, depthImg, outputFrame, frameInd);
			}
		}

	}
	if (curPoints.size() < kltPtThr)
	{
	}
	prevPoints = curPoints;
	prevImg = m_nextImg;
}

void Tracker::trackWithOrb(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd)
{

	
	orb->detect(m_nextImg, m_nextKeypoints);

	orb->compute(m_nextImg, m_nextKeypoints, m_nextDescriptors);

	//std::cout << m_nextKeypoints.size() << std::endl;
	if (m_prevKeypoints.size() > 0)
	{
		prevPoints = curPoints;
		curPoints.clear();
		//std::cout << prevPoints.size() << std::endl;

		std::vector< std::vector<cv::DMatch>>  matches;		
		m_orbMatcher->radiusMatch(m_nextDescriptors, m_prevDescriptors, matches, 30.0);

		for (size_t i = 0; i < matches.size(); i++)
		{
			if (matches[i].size() == 0)
			{
				createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
				continue;
			}
						
			cv::Point prevPt = m_prevKeypoints[matches[i][0].trainIdx].pt;
			cv::Point nextPt = m_nextKeypoints[matches[i][0].queryIdx].pt;

			cv::circle(outputFrame, prevPt, 5, cv::Scalar(250, 0, 250), -1);
			cv::line(outputFrame, prevPt, nextPt, cv::Scalar(0, 250, 0));
			cv::circle(outputFrame, nextPt, 3, cv::Scalar(0, 250, 0), -1);
			

			Track* foundTrack = prevPoints[matches[i][0].trainIdx];
			//std::cout << foundTrack->history.size() << std::endl;
			if (foundTrack->bestCandidate->frameId < frameInd || foundTrack->bestCandidate->matchScore > matches[i][0].distance)
			{
				foundTrack->bestCandidate = new TrackedPoint(nextPt, frameInd, matches[i][0].distance, m_nextKeypoints[matches[i][0].queryIdx], m_nextDescriptors.row(i));
			}						
		}
		
		for (int i = 0; i < prevPoints.size(); i++)
		{
			int lastHInd = prevPoints[i]->history.size();
			if (prevPoints[i]->bestCandidate->frameId < frameInd  && lastHInd > 3)
			{				
				lostTracks.push_back(prevPoints[i]);
			}
			if (prevPoints[i]->bestCandidate->frameId == frameInd)
			{
				//std::cout << prevPoints[i]->history.size() << std::endl;
				prevPoints[i]->history.push_back(prevPoints[i]->bestCandidate);
				//std::cout << prevPoints[i]->history.size() << std::endl;
				curPoints.push_back(prevPoints[i]);				
			}
		}
		m_prevDescriptors = cv::Mat(curPoints.size(), m_nextDescriptors.cols, m_nextDescriptors.type());
		m_prevKeypoints.clear();
		for (int i = 0; i < curPoints.size(); i++)
		{			
			curPoints[i]->bestCandidate->desc.copyTo(m_prevDescriptors.row(i));

			//std::cout << curPoints[i]->bestCandidate.keyPt.pt << std::endl;
			//std::cout << m_prevDescriptors.row(i) << std::endl;
			m_prevKeypoints.push_back(curPoints[i]->bestCandidate->keyPt);
		}
	}
	else {
		for (int i = 0; i < m_nextKeypoints.size(); i++)
		{
			createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
		}
		m_prevKeypoints.swap(m_nextKeypoints);
		m_nextDescriptors.copyTo(m_prevDescriptors);
	}
	
	
}

void Tracker::saveAllTracks(std::string& pathToSaveFolder)
{
	for (int i = 0; i < prevPoints.size(); i++)
	{
		lostTracks.push_back(prevPoints[i]);
	}
	std::cout << "final size " << lostTracks.size() << std::endl;
	for (int i = 0; i < lostTracks.size(); i++)
	{
		std::ofstream outTrackSave(pathToSaveFolder + std::to_string(i) + ".txt");
		Track* curTrack = lostTracks[i];
		for (int hInd = 0; hInd < curTrack->history.size(); hInd++)
		{
			outTrackSave << curTrack->history[hInd]->frameId << " " << curTrack->history[hInd]->location.x << " " << curTrack->history[hInd]->location.y << " " << curTrack->history[hInd]->depth << std::endl;
		}
	}
}
