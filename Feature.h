#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/core.hpp>
#include <vector>
#include <iostream>

// Class describing an interesting feature (corner) in an image
class Feature
{
private:
	static int id_gen;
public:
	// Feature extractor that found the feature
	const enum extractor { shi_tomasi, cv_good };

	int row = 0;
	int column = 0;
	extractor detector;
	bool tracked = true;
	double score = 0;
	double displacement = 0;

	int id;

	Feature(int column, int row) : row(row), column(column) { id = id_gen++; }
	Feature(int column, int row, extractor extr) : row(row), column(column), detector(extr) { id = id_gen++; }
	Feature(cv::Point p) : row(p.y), column(p.x) { id = id_gen++; }
	Feature() { tracked = false; }

	/**
		Creates an OpenCV point based on this feature

		@return OpenCV point with this feature's coordinates
	*/
	cv::Point point();

	cv::Point2f getPoint() { return cv::Point2f(column, row); }

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
	bool hasNeighbor(const std::vector<Feature> feats, int dist = 5);

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
