#pragma once

#include "stdafx.h"

#include "TrackedPoint.h"

enum PointType {Undef, Static, Dynamic};

class Track
{
public:
  Track();

  PointType type;
  float angle;
  float c, median;
  float err1;
  float err2;
  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::shared_ptr<TrackedPoint> bestCandidate;
};

