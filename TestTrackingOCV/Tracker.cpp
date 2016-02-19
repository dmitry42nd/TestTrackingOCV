#include "stdafx.h"

#include "Tracker.h"

#include "opencv2/video/tracking.hpp"

Tracker::Tracker(TrajectoryArchiver &trajArchiver) : trajArchiver(trajArchiver)
{
	orb = new cv::ORB(1000, 1.2, 8, 51);
	m_orbMatcher = new cv::BFMatcher(cv::NORM_HAMMING);

	fastDetector = new cv::FastFeatureDetector(10);
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

void Tracker::createNewTrack(cv::Point2f point, int frameCnt, cv::KeyPoint const & keyPt, cv::Mat const & desc, double depth)
{
  std::shared_ptr<Track> newTrack(std::make_shared<Track>());
	newTrack->history.push_back(std::make_shared<TrackedPoint>(point, frameCnt, 0, keyPt, desc, depth));
  newTrack->bestCandidate = 0;
	curPoints.push_back(newTrack);
}

void roundCoords(int &px, int &py, cv::Point2f const& pt, cv::Mat const& img)
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

cv::Mat Tracker::calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks)
{
  //num of points in each quadrant
	cv::Mat res = cv::Mat::zeros(wy, wx, CV_32F);
	int sx = imSize.width / wx;
	int sy = imSize.height/ wy;
  //count points in each quadrant
  //for (int i = 0; i < curTracks.size(); i++)
  for (auto const& c : curTracks)
	{
    auto const& bc = c->history[c->bestCandidate];
    int resX = bc->location.x / sx;
    int resY = bc->location.y / sy;
		res.at<float>(resY, resX) = res.at<float>(resY, resX) + 1;
	}
	cv::Mat boolRes = cv::Mat::zeros(wy, wx, CV_8U);
  //check if less than threshold ptNum
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

std::vector<cv::KeyPoint> Tracker::filterPoints(int wx, int wy, std::vector<cv::KeyPoint>& keyPts) {
  typedef std::pair<int, int> Coords;
  std::map<Coords, cv::KeyPoint> keyPtsMap;
  
  for (auto const& keyPt : keyPts) {
    int cx = keyPt.pt.x / wx;
    int cy = keyPt.pt.y / wy;

    if (keyPt.response > keyPtsMap[Coords(cx, cy)].response) {
      keyPtsMap[Coords(cx, cy)] = keyPt;
    }
  }

  //copy map to vector as needed
  std::vector<cv::KeyPoint> keyPtsNew;
  for (auto const& keyPt : keyPtsMap) {
    keyPtsNew.push_back(keyPt.second);
  }

  return keyPtsNew;
}

void Tracker::detectPoints(int indX, int indY, cv::Mat& m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd) {

	std::vector<cv::KeyPoint> keyPts;
	std::cout << " detecting.. " << indX << " " << indY << std::endl;
	fastDetector->detect(m_nextImg, keyPts, detMasks[indY][indX]);
  int keyPtsSize = keyPts.size();
  //draw all FAST points (blue)
  //for (int i = 0; i < keyPts.size(); i++)
  for (auto const& kp: keyPts) {
    cv::circle(outputFrame, kp.pt, 3, cv::Scalar(255, 0, 0));
	}

#if 1
  //get 80% of best by reduction points
  cv::KeyPointsFilter::retainBest(keyPts, keyPts.size() * 6 / 10); 
  //draw all filtered points (yellow)
  for (auto const& kp : keyPts) {
    cv::circle(outputFrame, kp.pt, 3, cv::Scalar(0, 255, 255));
  }
#endif

  //TODO: magic numbers 16, 16
  std::vector<cv::KeyPoint> keyPtsFiltered = filterPoints(16, 16, keyPts);
  auto repProc = keyPtsFiltered.size() * 100 / keyPtsSize;
  std::cout << "key point reduction: " << repProc << " % " << keyPtsSize << ":" << keyPtsFiltered.size() << "\n";

  //draw final filtered points (red) and some stuff
  for (int i = 0; i < keyPtsFiltered.size(); i++) {
    int px = 0, py = 0;

    cv::Point2f pt = fillDepthPt(keyPtsFiltered[i].pt, ccx, ccy, cfx, cfy, dcx, dcy, dfx, dfy);
    roundCoords(px, py, pt, m_nextImg);
    double depthVal = (int)(depthImg.at<ushort>(py, px) / 5000.0);
    createNewTrack(keyPtsFiltered[i].pt, frameInd, keyPtsFiltered[i], cv::Mat(), depthVal);

    cv::circle(outputFrame, keyPtsFiltered[i].pt, 3, cv::Scalar(0, 0, 255));
  }
}

void Tracker::trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg) {

	curPoints.clear();

	cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);

  if (prevPoints.size() > 0) {    
		std::vector<cv::Point2f> prevCorners;
    for(auto &p : prevPoints) {
      prevCorners.push_back(p->history.back()->location);
		}

		std::vector<cv::Point2f> nextCorners;
		std::vector<uchar> status;
		std::vector<float> err;
		double minEigThreshold = 1e-2;
		cv::calcOpticalFlowPyrLK(prevImg, m_nextImg, prevCorners, nextCorners, status, err,
			                       cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
			                       cv::OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold);

    for (size_t i = 0; i < prevPoints.size(); i++) {
			cv::Mat err = cv::Mat(nextCorners[i] - prevCorners[i]);
			double trackDist = norm(err);
			if (trackDist < trackThr && status[i] && nextCorners[i].x >= 0 && nextCorners[i].x < m_nextImg.cols &&
        nextCorners[i].y >= 0 && nextCorners[i].y < m_nextImg.rows) {
				//std::cout << "track depth read " << round(nextCorners[i].y) << " " << round(nextCorners[i].x) << std::endl;
				//std::cout << nextCorners[i] << std::endl;
				cv::Point2f pt = fillDepthPt(nextCorners[i], ccx, ccy, cfx, cfy, ccx, ccy, cfx, cfy);
				int px, py;
				
				//std::cout << pt<< std::endl;
				roundCoords(px, py, pt, m_nextImg);
				double depthVal = 0;
        if (pt.x > 0 && pt.y > 0 && pt.x < depthImg.cols && pt.y < depthImg.rows) {
					depthVal = (int)depthImg.at<ushort>(py, px);
					depthVal /= 5000.0;
				}
        prevPoints[i]->bestCandidate = prevPoints[i]->history.size();
        prevPoints[i]->history.push_back(std::make_shared<TrackedPoint>(nextCorners[i], frameInd, 0, cv::KeyPoint(), cv::Mat(), depthVal));
				curPoints.push_back(prevPoints[i]);

				cv::circle(outputFrame, prevCorners[i], 5, cv::Scalar(250, 0, 250), -1);
				cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
				cv::circle(outputFrame, nextCorners[i], 3, cv::Scalar(0, 250, 0), -1);
			}
      else {
        if (prevPoints[i]->history.size() > 3) {
					lostTracks.push_back(prevPoints[i]);
					trajArchiver.archiveTrajectory(prevPoints[i]);
				}				
			}
		}
	}

	cv::Mat boolStats = calcStatsByQuadrant(wx, wy, kltPtThr / 4, curPoints);

  for (int i = 0; i < wx; i++) {
    for (int j = 0; j < wy; j++) {
      if (boolStats.at<uchar>(j, i) > 0) {
				detectPoints(i, j, m_nextImg, depthImg, outputFrame, frameInd);
			}
		}
  }

  if (curPoints.size() < kltPtThr) {  }

	prevPoints = curPoints;
	prevImg = m_nextImg;
}

#if 1
void Tracker::trackWithOrb(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd)
{
	orb->detect(m_nextImg, m_nextKeypoints);
	orb->compute(m_nextImg, m_nextKeypoints, m_nextDescriptors);

  //std::cout << "next keypoints: " << m_nextKeypoints.size() << std::endl;
  if (m_prevKeypoints.size() > 0) {
    prevPoints.swap(curPoints);
		curPoints.clear();

    std::vector<std::vector<cv::DMatch>>  matches;    
		m_orbMatcher->radiusMatch(m_nextDescriptors, m_prevDescriptors, matches, 30.0);
    //std::cout << "mathces: " << matches.size() << std::endl;

    for (size_t i = 0; i < matches.size(); i++) {
      if (matches[i].size() == 0) {
				createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
				continue;
			}
						
			cv::Point prevPt = m_prevKeypoints[matches[i][0].trainIdx].pt;
			cv::Point nextPt = m_nextKeypoints[matches[i][0].queryIdx].pt;

			cv::circle(outputFrame, prevPt, 5, cv::Scalar(250, 0, 250), -1);
			cv::line(outputFrame, prevPt, nextPt, cv::Scalar(0, 250, 0));
			cv::circle(outputFrame, nextPt, 3, cv::Scalar(0, 250, 0), -1);
			
			std::shared_ptr<Track> foundTrack = prevPoints[matches[i][0].trainIdx];
			//std::cout << foundTrack->history.size() << std::endl;
      auto bc = foundTrack->bestCandidate;
      if (foundTrack->history[bc] ->frameId < frameInd || foundTrack->history[bc]->matchScore > matches[i][0].distance) {
        foundTrack->history[bc] = std::make_shared<TrackedPoint>(nextPt, frameInd, matches[i][0].distance, m_nextKeypoints[matches[i][0].queryIdx], m_nextDescriptors.row(i));
			}						
		}
		
    for (int i = 0; i < prevPoints.size(); i++) {
			int lastHInd = prevPoints[i]->history.size();
      auto bc = prevPoints[i]->bestCandidate;
      if (prevPoints[i]->history[bc]->frameId < frameInd  && lastHInd > 3) {
				lostTracks.push_back(prevPoints[i]);
			}

      if (prevPoints[i]->history[bc]->frameId == frameInd) {
				//std::cout << prevPoints[i]->history.size() << std::endl;
        prevPoints[i]->history.push_back(prevPoints[i]->history[bc]);
				//std::cout << prevPoints[i]->history.size() << std::endl;
				curPoints.push_back(prevPoints[i]);				
			}
		}

		m_prevDescriptors = cv::Mat(curPoints.size(), m_nextDescriptors.cols, m_nextDescriptors.type());
		m_prevKeypoints.clear();

    for (int i = 0; i < curPoints.size(); i++) {
      auto bc = curPoints[i]->bestCandidate;
      curPoints[i]->history[bc]->desc.copyTo(m_prevDescriptors.row(i));

			//std::cout << curPoints[i]->bestCandidate.keyPt.pt << std::endl;
			//std::cout << m_prevDescriptors.row(i) << std::endl;
      m_prevKeypoints.push_back(curPoints[i]->history[bc]->keyPt);
	}
  } else {
    for (int i = 0; i < m_nextKeypoints.size(); i++) {
			createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
		}
		m_prevKeypoints.swap(m_nextKeypoints);
		m_nextDescriptors.copyTo(m_prevDescriptors);
	}
}
#endif

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
		std::shared_ptr<Track> curTrack = lostTracks[i];
    for (int hInd = 0; hInd < curTrack->history.size(); hInd++) {
			outTrackSave << curTrack->history[hInd]->frameId << " " << curTrack->history[hInd]->location.x << " " << curTrack->history[hInd]->location.y << " " << curTrack->history[hInd]->depth << std::endl;
		}
	}
}