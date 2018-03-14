#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <memory>

// Class describing an interesting feature (corner) in an image
class Feature
{
public:
	// Feature extractor that found the feature
	enum extractor { shi_tomasi, cv_good };

	int row = 0;
	int column = 0;
	extractor detector;
	bool tracked = true;
	double score = 0;
	double displacement = 0;

	Feature(int column, int row) : row(row), column(column) { }
	Feature(int column, int row, extractor extr) : Feature(row, column) { detector = extr; }
	Feature(cv::Point p) : Feature(p.x, p.y) { }
	Feature() { tracked = false; }

	struct Hasher
	{
		std::size_t operator()(const std::weak_ptr<Feature> f) const
		{
			if (f.expired())
				return 0;
			std::shared_ptr<Feature> f_ptr = f.lock();
			size_t const h1(std::hash<std::string>{}(std::to_string(f_ptr->column)));
			size_t const h2(std::hash<std::string>{}(std::to_string(f_ptr->row)));

			return h1 ^ (h2 << 1);
		}

		std::size_t operator()(const std::shared_ptr<Feature> f) const
		{
			size_t const h1(std::hash<std::string>{}(std::to_string(f->column)));
			size_t const h2(std::hash<std::string>{}(std::to_string(f->row)));

			return h1 ^ (h2 << 1);
		}
	};

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

	friend bool operator== (const Feature& lhs, const Feature& rhs);
	friend bool operator!= (const Feature& lhs, const Feature& rhs);
	friend bool operator>(const Feature& lhs, const Feature& rhs);

	friend bool operator== (const std::weak_ptr<Feature> lhs, const std::weak_ptr<Feature> rhs);

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
