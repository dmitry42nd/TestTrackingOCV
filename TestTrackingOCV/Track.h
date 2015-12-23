#pragma once
#include <vector>
#include "TrackedPoint.h"
class Track
{
public:
	Track();
	~Track();

	std::vector<TrackedPoint*> history;
	TrackedPoint *bestCandidate;

};

