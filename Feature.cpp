#include <opencv2/core.hpp>
#include "Feature.h"

cv::Point Feature::point()
{
	return cv::Point(column, row);
}

float Feature::distance(const Feature& f)
{
	//return sqrt(pow(f.column - column, 2) + pow(f.row - row, 2));
	int x = abs(column - f.column);
	int y = abs(row - f.row);
	return x > y ? x : y;
}

void Feature::scale(float scale)
{
	column = (int)((float)column*scale);
	row = (int)((float)row*scale);
}

void Feature::scale(std::vector<Feature>& feats, float scale)
{
	for (auto & f : feats)
	{
		f.scale(scale);
	}
}

bool operator==(const Feature& lhs, const Feature& rhs)
{
	if (lhs.row == rhs.row && lhs.column == rhs.column)
		return true;
	return false;
}

bool operator!=(const Feature& lhs, const Feature& rhs)
{
	return !(lhs == rhs);
}

bool operator>(const Feature& lhs, const Feature& rhs)
{
	return (lhs.score > rhs.score);
}

bool operator==(const std::weak_ptr<Feature> lhs, const std::weak_ptr<Feature> rhs)
{
	if (lhs.expired() || rhs.expired())
		return false;
	std::shared_ptr<Feature> lhs_ptr = lhs.lock();
	std::shared_ptr<Feature> rhs_ptr = rhs.lock();
	return (lhs_ptr->column == rhs_ptr->column && lhs_ptr->row == rhs_ptr->row);
}
