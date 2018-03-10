#ifndef CERES_BUNDLE_ADJUSTMENT_H
#define CERES_BUNDLE_ADJUSTMENT_H
#include "Tracker.h"

class CeresBundleAdjustment : public BaseOptimizer
{
public:
	Tracker* tracker;

	CeresBundleAdjustment(Tracker* tracker) : tracker(tracker) {}

	void apply(Frame& f);
};

#endif