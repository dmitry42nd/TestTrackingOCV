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
  float err[3];

  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::shared_ptr<TrackedPoint> bestCandidate;
};

