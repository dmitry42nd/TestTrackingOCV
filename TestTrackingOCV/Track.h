#pragma once
#include <vector>
#include <memory>
#include "TrackedPoint.h"

class Track
{
public:
  Track();

  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::shared_ptr<TrackedPoint> bestCandidate;
};

