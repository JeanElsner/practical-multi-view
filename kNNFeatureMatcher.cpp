#include "kNNFeatureMatcher.h"

BaseFeatureMatcher::fmap kNNFeatureMatcher::matchFeatures(Frame & src, Frame & next)
{
	fmap map;
	double avg = 0;
	std::vector<std::shared_ptr<Feature>> new_feats, old_feats;
	std::vector<bool> tracked;

	std::vector<Feature> cmp_feats = extractor->extractFeatures(next, 1000);

	for (auto& p : src.map)
		old_feats.push_back(p.first);

	for (auto& f: old_feats)
	{
		std::vector<Feature> nn = getNearestNeighbors(*f, cmp_feats);
		Feature best_fit;
		float err = 0;

		for (auto const& ff : nn)
		{
			float _err = compareFeatures(src.bw, f->column, f->row, next.bw, ff.column, ff.row);

			if (_err < err || err == 0)
			{
				err = _err;
				best_fit = ff;
			}
		}
		if (err < threshold)
		{
			tracked.push_back(true);
			best_fit.tracked = true;
			best_fit.displacement = f->distance(best_fit);
			avg += best_fit.displacement;
		}
		else
			tracked.push_back(false);
		new_feats.push_back(std::make_shared<Feature>(best_fit));
	}
	avg /= (double)new_feats.size();

	for (auto& f : new_feats)
	{
		if (f->displacement > 3 * avg)
		{
			f->tracked = false;
		}
	}

	for (int i = 0; i < old_feats.size(); i++)
	{
		if (new_feats[i]->tracked)
		{
			next.map[new_feats[i]] = src.map[old_feats[i]];
			map[old_feats[i]] = new_feats[i];
		}
	}
	return map;
}

std::vector<Feature> kNNFeatureMatcher::getNearestNeighbors(
	Feature& f, const std::vector<Feature>& feats, int n)
{
	std::vector<Feature> nearest_vec;
	Feature nearest;

	for (int k = 0; k < n; k++)
	{
		float dist = 0;

		for (auto const& ff : feats)
		{
			if (f != ff)
			{
				bool b = true;

				for (auto const& fff : nearest_vec)
				{
					if (ff == fff)
						b = false;
				}
				if (b)
				{
					float _dist = f.distance(ff);

					if (_dist < dist || dist == 0)
					{
						dist = _dist;
						nearest = ff;
					}
				}
			}
		}
		nearest_vec.push_back(nearest);
	}
	return nearest_vec;
}

float kNNFeatureMatcher::compareFeatures(
	const cv::Mat& src, int src_x, int src_y, const cv::Mat& cmp, int cmp_x, int cmp_y)
{
	int _win = ceil((float)window / 2.f);
	float err = 0;

	for (schar x = -_win; x < _win + 1; x++)
	{

		for (schar y = -_win; y < _win + 1; y++)
		{
			if (src_x + x < 0 || src_y + y < 0 || cmp_x + x < 0 || cmp_y + y < 0
				|| src_x + x >= src.cols || src_y + y >= src.rows 
				|| cmp_x + x >= cmp.cols || cmp_y + y >= cmp.rows)
				continue;
			err += pow((float)src.at<uchar>(cv::Point(src_x + x, src_y + y)) 
				- (float)cmp.at<uchar>(cv::Point(cmp_x + x, cmp_y + y)), 2);
		}
	}
	err = sqrt(err) / (pow(window, 2));
	return err;
}