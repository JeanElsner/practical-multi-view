#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "Feature.h"
#include "Frame.h"
#include "BaseFeatureExtractor.h"
#include "OpenCVGoodFeatureExtractor.h"
#include "ShiTomasiFeatureExtractor.h"
#include "OpenCVLucasKanadeFM.h"
#include "Tracker.h"

using namespace cv;
using namespace std;

unsigned char GRID_SIZE[2] = { 255, 255 };

std::vector<double> ticktock;

/**
	Starts a timer, will be tiered if called multiple times
*/
void tick()
{
	ticktock.push_back(getTickCount());
}

/**
	Returns the time since the last tick in seconds

	@return Time since last call to tick
*/
double tock()
{
	if (ticktock.empty())
		return 0;
	double tock = ticktock.back();
	ticktock.pop_back();
	return (getTickCount() - tock) / getTickFrequency();
}

/**
	Divides the image into a grid, so as to distribute
	the calculated features across the entire image.

	@param I Input image
	@param size The size of the grid blocks in pixels
	@param fn The number of features per block
	@param extr Vector of functions used to extract features
*/
vector<Feature> grid_feature_extraction(Frame& I, const unsigned char size[2], const int fn,
	vector<BaseFeatureExtractor*> extr)
{
	int gr = I.bw.rows / size[0];
	int gc = I.bw.cols / size[1];
	vector<Feature> feats;

	for (int r = 0; r < I.bw.rows; r += size[0])
	{
		for (int c = 0; c < I.bw.cols; c += size[1])
		{
			Rect rec(c, r, min((int)size[1], I.bw.cols - c), min((int)size[0], I.bw.rows - r));
			Frame roi = I.regionOfInterest(rec);
			
			for (auto& e : extr)
			{
				int count = 0;
				vector<Feature> new_feats = e->extractFeatures(roi, fn);
				for (auto& f : new_feats)
				{
					f.column = c + f.column;
					f.row = r + f.row;
					if (count >= fn)
						break;
					if (!f.hasNeighbor(feats))
					{
						count++;
						feats.push_back(f);
					}
				}
			}
		}
	}
	return feats;
}

// TODO: doc
void gaussian_window_3x3(const Mat& src, Mat& dst)
{
	Mat kernel = (Mat_<double>(3, 3) << 1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16);
	filter2D(src, dst, -1, kernel);
	return;

	for (int r = 1; r < src.rows - 1; r++)
	{
		const double* prev = src.ptr<double>(r - 1);
		const double* curr = src.ptr<double>(r);
		const double* next = src.ptr<double>(r + 1);
		float* p_dst = dst.ptr<float>(r);

		for (int c = 1; c < src.cols - 1; c++)
		{
			p_dst[c] = 1. / 8. * (prev[c - 1] + prev[c + 1] + next[c - 1] + next[c + 1])
				+ 1. / 16. * (prev[c] + curr[c - 1] + curr[c + 1] + next[c]) + 1. / 4. * (curr[c]);
		}
	}
}

// TODO: doc
uchar bilinear_subpixel(const Mat& src, float sub_x, float sub_y)
{
	int x = (int)sub_x;
	int y = (int)sub_y;
	float a = sub_x - (float)x;
	float c = sub_y - (float)y;

	int x0 = borderInterpolate(x, src.cols, cv::BORDER_REFLECT_101);
	int y0 = borderInterpolate(y, src.rows, cv::BORDER_REFLECT_101);
	int x1 = borderInterpolate(x + 1, src.cols, cv::BORDER_REFLECT_101);
	int y1 = borderInterpolate(y + 1, src.rows, cv::BORDER_REFLECT_101);

	return (uchar)cvRound((src.at<char>(y0, x0) * (1.f - a) + src.at<char>(y0, x1) * a) * (1.f - c)
		+ (src.at<char>(y1, x0) * (1.f - a) + src.at<char>(y1, x1) * a) * c);
}

// TODO: doc
void track_features(Frame& f_src, const Frame& f_next, const vector<Feature>& feats, vector<Feature>& new_feats, int window = 5, int iter = 5)
{
	vector<Mat> grad;
	grad.push_back(f_src.getSpatialGradientX());
	grad.push_back(f_src.getSpatialGradientY());
	
	int _win = ceil((float)window / 2.f);

	for (auto const& f : feats)
	{
		double v[2] = { .0, .0 };
		Mat G = Mat::zeros(2, 2, CV_64FC1);

		for (schar x = -_win; x < _win+1; x++)
		{
			for (schar y = -_win; y < _win+1; y++)
			{
				if (f.column + x < 0 || f.column + x >= f_src.bw.cols
					|| f.row + y < 0 || f.row + y >= f_src.bw.rows)
					continue;
				G.at<double>(Point(0, 0)) += pow(grad[0].at<double>(Point(f.column + x, f.row + y)), 2);
				G.at<double>(Point(1, 1)) += pow(grad[1].at<double>(Point(f.column + x, f.row + y)), 2);
				G.at<double>(Point(1, 0)) += (grad[0].at<double>(Point(f.column + x, f.row + y))
					* grad[1].at<double>(Point(f.column + x, f.row + y)));
				G.at<double>(Point(0, 1)) += (grad[0].at<double>(Point(f.column + x, f.row + y))
					* grad[1].at<double>(Point(f.column + x, f.row + y)));
				/*G.at<double>(Point(0, 0)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[0];
				G.at<double>(Point(1, 1)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[1];
				G.at<double>(Point(1, 0)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[2];
				G.at<double>(Point(0, 1)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[2];*/
			}
		}
		Mat b = Mat::zeros(2, 1, CV_64FC1);
		Mat err;
		for (size_t j = 0; j < iter; j++)
		{
			for (schar x = -_win; x < _win+1; x++)
			{
				for (schar y = -_win; y < _win+1; y++)
				{
					if (f.column + x + v[0] < 0 || f.column + x + v[0] + 1 > f_src.bw.cols
						|| f.row + y + v[1] < 0 || f.row + y + v[1] + 1 > f_src.bw.rows)
						continue;
					if (f.column + x < 1 || f.column + x + 1 >= f_next.bw.cols
						|| f.row + y < 1 || f.row + y + 1 >= f_next.bw.rows)
						continue;

					/*double Ix = (double)src.at<uchar>(Point(f.column + x + 1, f.row + y))
						- (double)src.at<uchar>(Point(f.column + x - 1, f.row + y));
					double Iy = (double)src.at<uchar>(Point(f.column + x, f.row + y  + 1))
						- (double)src.at<uchar>(Point(f.column + x, f.row + y - 1));
					Ix *= 1. / 2.;
					Iy *= 1. / 2.;*/

					b.at<double>(Point(0, 0)) += (((double)f_src.bw.at<uchar>(Point(f.column + x, f.row + y)) -
						//(double)next.at<uchar>(Point(f.column + x + v[0], f.row + y + v[1]))) * 
						(double)bilinear_subpixel(f_next.bw, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) *
						grad[0].at<double>(Point(f.column + x, f.row + y)));
						//(double)bilinear_subpixel(next, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) * Ix;
						//sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[0]);
						//H.at<Vec3f>(Point(f.column + x, f.row + y))[0];
					b.at<double>(Point(0, 1)) += (((double)f_src.bw.at<uchar>(Point(f.column + x, f.row + y)) -
						//(double)next.at<uchar>(Point(f.column + x + v[0], f.row + y + v[1]))) *
						(double)bilinear_subpixel(f_next.bw, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) *
						grad[1].at<double>(Point(f.column + x, f.row + y)));
						//(double)bilinear_subpixel(next, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) * Iy;
						//sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[1]);sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[1]);
						//H.at<Vec3f>(Point(f.column + x, f.row + y))[1];
				}
			}
			Mat G_inv = Mat::zeros(Size(2, 2), G.type());
			double det = G.at<double>(Point(0, 0))*G.at<double>(Point(1, 1))
				- pow(G.at<double>(Point(1, 0)), 2);
			
			if (det != 0)
			{
				G_inv.at<double>(Point(0, 0)) = 1. / det*G.at<double>(Point(1, 1));
				G_inv.at<double>(Point(1, 0)) = -1. / det*G.at<double>(Point(1, 0));
				G_inv.at<double>(Point(0, 1)) = -1. / det*G.at<double>(Point(0, 1));
				G_inv.at<double>(Point(1, 1)) = 1. / det*G.at<double>(Point(0, 0));
			}
			err = G_inv*b;
			
			v[0] += err.at<double>(Point(0, 0));
			v[1] += err.at<double>(Point(0, 1));
		}
		Feature corr;
		if ((pow(err.at<double>(Point(0, 0)), 2) + pow(err.at<double>(Point(0, 1)), 2)) > 1)
			corr.tracked = false;
		else
			corr.tracked = true;
		corr.column = (int)round((double)f.column + v[0]);
		corr.row = (int)round((double)f.row + v[1]);
		new_feats.push_back(corr);
	}
}

// TODO: doc
vector<Feature> knn_features(Feature& f, const vector<Feature>& feats, int n = 7)
{
	vector<Feature> nearest_vec;
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

// TODO: doc
float compare_features(const Mat& src, int src_x, int src_y, const Mat& cmp, int cmp_x, int cmp_y, int window)
{
	int _win = ceil((float)window / 2.f);
	float err = 0;

	for (schar x = -_win; x < _win + 1; x++)
	{

		for (schar y = -_win; y < _win + 1; y++)
		{
			if (src_x + x < 0 || src_y + y < 0 || cmp_x + x < 0 || cmp_y + y < 0
				|| src_x + x >= src.cols || src_y + y >= src.rows || cmp_x + x >= cmp.cols || cmp_y + y >= cmp.rows)
				continue;
			err += pow((float)src.at<uchar>(Point(src_x + x, src_y + y)) - (float)cmp.at<uchar>(Point(cmp_x + x, cmp_y + y)), 2);
		}
	}
	err = sqrt(err)/(pow(window, 2));
	return err;
}

// TODO: doc
void knn_tracker(const Frame& f_src, Frame& f_cmp, vector<Feature> feats, 
	vector<Feature>& new_feats, vector<bool>& tracked, int window = 15, float threshold = 2)
{
	vector<BaseFeatureExtractor*> funcs;
	//funcs.push_back(shi_tomasi_detector);
	funcs.push_back(&OpenCVGoodFeatureExtractor());

	vector<Feature> cmp_feats = grid_feature_extraction(f_cmp, GRID_SIZE, 25, funcs);
	double avg = 0;
	for (auto & f : feats)
	{
		vector<Feature> nn = knn_features(f, cmp_feats);
		Feature best_fit;
		float err = 0;

		for (auto const& ff : nn)
		{
			float _err = compare_features(f_src.bw, f.column, f.row, f_cmp.bw, ff.column, ff.row, window);

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
			best_fit.displacement = f.distance(best_fit);
			avg += best_fit.displacement;
		}
		else
			tracked.push_back(false);
		new_feats.push_back(best_fit);
	}
	avg /= (double)new_feats.size();
	for (auto& f : new_feats)
	{
		if (f.displacement > 3*avg)
		{
			f.tracked = false;
			cout << avg << " > " << f.displacement << endl;
		}
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
		return -1;
	cv::String path(argv[1]);
	vector<cv::String> fn;
	cv::glob(path, fn, true);

	vector<Frame> I;
	vector<Mat> G;
	vector<Feature> feats;

	VideoWriter video;

	BaseFeatureMatcher* matcher = &OpenCVLucasKanadeFM();
	Tracker tracker();

	tick();
	for (size_t k = 0; k < fn.size(); ++k)
	{
		//if (k > 2) break;

		Frame frame(fn[k]);
		
		if (frame.isEmpty())
			continue;

		I.push_back(frame);
		
		vector<BaseFeatureExtractor*> funcs;
		funcs.push_back(&OpenCVGoodFeatureExtractor());
		//funcs.push_back(&ShiTomasiFeatureExtractor());

		/*
		Mat H(im.size(), CV_32FC3);
		compute_harris_matrix(im, H);
		G.push_back(H);
		feats = grid_feature_extraction(im, H, GRID_SIZE, 50, funcs);

		for (auto & f : feats)
		{
			circle(imc, f.point(), 5, Scalar(0, 255, 255));
		}
		imshow("test", imc);
		waitKey(20);*/
		
		// TODO: put these three section into trackers
		
		if (k == 0)
		{
			Mat H = frame.getHarrisMatrix();
			G.push_back(H);
			feats = grid_feature_extraction(frame, GRID_SIZE, 25, funcs);

			video = VideoWriter("../../my-feats/tracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, frame.bw.size());
		}
		
		//////////////////////////////
		// My knn tracker
		//////////////////////////////
		
		/*if (k > 0)
		{
			vector<Feature> new_feats;
			vector<bool> tracked;

			knn_tracker(I[k - 1], I[k], feats, new_feats, tracked);

			for (int i = 0; i < new_feats.size(); i++)
			{
				if (!new_feats[i].tracked)
					continue;

				circle(frame.orig, new_feats[i].point(), 5, Scalar(0, 255, 255));
				line(frame.orig, feats[i].point(), new_feats[i].point(), Scalar(0, 255, 255));
			}
			feats = grid_feature_extraction(frame, GRID_SIZE, 35, funcs);

			imshow("test", frame.orig);
			waitKey(20);
			video.write(frame.orig);
		}*/
		

		////////////////////////////////
		// Opencv lukas-kanade tracker
		////////////////////////////////
		
		if (k > 0)
		{
			std::vector<Feature> new_feats;
			matcher->matchFeatures(I[k - 1], I[k], feats, new_feats);

			for (int i = 0; i < new_feats.size(); i++)
			{
				if (new_feats[i].tracked)
				{
					circle(frame.orig, new_feats[i].point(), 5, Scalar(0, 255, 255));
					line(frame.orig, feats[i].point(), new_feats[i].point(), Scalar(0, 255, 255));
				}
			}
			feats = new_feats;
		}
		imshow("test", frame.orig);
		video.write(frame.orig);
		waitKey(20);
		
		
		//////////////////////////////////
		// my klt tracker
		//////////////////////////////////
		/*
		G.push_back(frame.getHarrisMatrix());
		if (k > 0)
		{
			vector<Feature> new_feats, old_feats;
			old_feats = feats;

			for (int j = 1; j >= 0; j--)
			{
				Mat src, next;//, H;
				next = Mat::zeros(I[k].bw.size() / 2, I[k].bw.type());

				resize(I[k].orig, next, I[k].orig.size() / (int)pow(2, j));
				resize(I[k-1].orig, src, I[k-1].orig.size() / (int)pow(2, j));
				//resize(G[k-1], H, G[k-1].size() / (int)pow(2, j));
				
				Feature::scale(feats, 1.f / pow(2, j));
				track_features(Frame(src), Frame(next), feats, new_feats, 31);
				
				Feature::scale(feats, pow(2, j));
				Feature::scale(new_feats, pow(2, j));

				if (j > 0)
				{
					feats = new_feats;
					new_feats.clear();
				}
			}
			Mat show = frame.orig.clone();
			for (int i = 0; i < new_feats.size(); i++)
			{
				if (!new_feats[i].tracked)
					continue;
				circle(show, Point(new_feats[i].column, new_feats[i].row), 5, Scalar(0, 255, 255));
				arrowedLine(show, new_feats[i].point(), old_feats[i].point(), Scalar(0, 255, 255));
			}
			feats.clear();

			for (auto const& f : new_feats)
			{
				feats.push_back(f);
			}
			imshow("test", show);
			waitKey(20);
		}
		*/
		/*for (auto const& f : feats)
		{
			switch (f.detector)
			{
			case DETECTOR_SHITOMASI:
				circle(imc, Point(f.column, f.row), 5, Scalar(0, 255, 255));
				break;
			case DETECTOR_CVGOOD:
				circle(imc, Point(f.column, f.row), 5, Scalar(0, 0, 255));
				break;
			}
		}*/
		stringstream f;
		f << "../../my-feats/" << k << ".png";
		string ff; f >> ff;
		cv::imwrite(ff, frame.orig);
	}
	video.release();
	cout << tock();
	waitKey();
	return 0;
}
