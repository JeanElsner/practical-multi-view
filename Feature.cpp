#include <opencv2/core.hpp>
#include "Feature.h"

cv::Point Feature::point()
{
	return cv::Point(column, row);
}

float Feature::distance(const Feature& f)
{
	return sqrt(pow(f.column - column, 2) + pow(f.row - row, 2));
}