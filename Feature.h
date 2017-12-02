#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/core.hpp>

class Feature
{
public:
	// Feature extractor that found the feature
	const enum extractor { shi_tomasi, cv_good };

	int row;
	int column;
	extractor detector;
	bool tracked;

	Feature(int column, int row) : row(row), column(column) { }
	Feature(int column, int row, extractor extr) : row(row), column(column), detector(extr) { }
	Feature() { }

	/**
		Creates an OpenCV point based on this feature

		@returns OpenCV point with this feature's coordinates
	*/
	cv::Point point();

	/**
		Calculates the distance from this to the given feature

		@param f The target feature
		@returns Distance between the features
	*/
	float distance(const Feature& f);
};
#endif