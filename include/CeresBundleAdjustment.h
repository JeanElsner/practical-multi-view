#ifndef CERES_BUNDLE_ADJUSTMENT_H
#define CERES_BUNDLE_ADJUSTMENT_H
#include "OdometryPipeline.h"

class CeresBundleAdjustment : public BaseOptimizer
{
public:
	OdometryPipeline* tracker;

	CeresBundleAdjustment(OdometryPipeline* tracker) : tracker(tracker) {}

	void apply(Frame& f);
};

#endif