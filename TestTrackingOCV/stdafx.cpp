// stdafx.cpp : source file that includes just the standard includes
// TestTrackingOCV.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

bool ifTracksEnd(int frameId)
{
  //  kinect 601
  //const int ends[5] = {653, 720, 782, 857, 917};
  //const int ends[5] = {347, 429, 499}; //uno_old
  const int ends[5] = {351, 433, 503}; //uno
  //const int ends[5] = {93};
  for(auto i : ends)
  {
    if(frameId == i) {
      std::cerr << "end of track\n";
      return true;
    }
  }

  return false;
}

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file
