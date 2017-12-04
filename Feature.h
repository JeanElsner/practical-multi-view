#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/core.hpp>
#include <vector>

// Class describing an interesting feature (corner) in an image
class Feature
{
public:
	// Feature extractor that found the feature
	const enum extractor { shi_tomasi, cv_good };

	int row;
	int column;
	extractor detector;
	bool tracked = false;
	double score = 0;
	double displacement = 0;

	Feature(int column, int row) : row(row), column(column) { }
	Feature(int column, int row, extractor extr) : row(row), column(column), detector(extr) { }
	Feature() { }

	/**
		Creates an OpenCV point based on this feature

		@return OpenCV point with this feature's coordinates
	*/
	cv::Point point();

	/**
		Calculates the distance from this to the given feature

		@param f The target feature
		@return Distance between the features
	*/
	float distance(const Feature& f);

	/**
		Checks whether this feature has any neighbors within
		a given (Manhattan) distance.

		@param feats A list of features to check against
		@param dist The distance in pixels
		@return True if a neigihbor was found, false otherwise
	*/
	bool hasNeighbor(const std::vector<Feature> feats, int dist = 3);

	friend bool operator== (const Feature& lhs, const Feature& rhs);
	friend bool operator!= (const Feature& lhs, const Feature& rhs);
	friend bool operator>(const Feature& lhs, const Feature& rhs);

	/**
		Scale this feature's coordinates by a factor

		@param scale Scale factor
	*/
	void scale(float scale);

	/**
		Scale the features' coordinates by a factor

		@param scale Scale Factor
	*/
	static void scale(std::vector<Feature>& feats, float scale);
};
#endif
