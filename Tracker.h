#ifndef TRACKER_H
#define TRACKER_H
#include "Feature.h";
#include "Frame.h"
#include <vector>

class Tracker
{
public:
	std::vector<Feature> features;
	std::vector<Frame> frames;
};
#endif
