#pragma once
#include <vector>
#include "boost/ptr_container/ptr_vector.hpp"
#include "TrackedPoint.h"
class Track
{
public:
	Track();
	~Track();

  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::shared_ptr<TrackedPoint> bestCandidate;
};

