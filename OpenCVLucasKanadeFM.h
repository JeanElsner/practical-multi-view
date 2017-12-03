#ifndef OPEN_CV_LUCAS_KANADE_FM_H
#define OPEN_CV_LUCAS_KANADE_FM_H
#include "BaseFeatureMatcher.h"

class OpenCVLucasKanadeFM :
	public BaseFeatureMatcher
{
public:

	virtual void matchFeatures(
		Frame& src,
		Frame& next,
		std::vector<Feature>& feats,
		std::vector<Feature>& new_feats
	);

};
#endif
