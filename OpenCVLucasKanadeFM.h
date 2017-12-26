#ifndef OPEN_CV_LUCAS_KANADE_FM_H
#define OPEN_CV_LUCAS_KANADE_FM_H
#include "BaseFeatureMatcher.h"

class OpenCVLucasKanadeFM :
	public BaseFeatureMatcher
{
private:
	int win_size = 21;
	int pyr_size = 4;

public:
	virtual fmap matchFeatures(Frame& src, Frame& next);

};
#endif
