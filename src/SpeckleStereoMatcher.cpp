/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-08
 */

#include "SpeckleStereoMatcher.hpp"
#include "SpeckleStereoHelper.hpp"
#include <omp.h>
#include <fstream>

using namespace SpeckleStereo;

inline double calc_xcorr(const cv::Mat &left, const cv::Mat &right, int x_start, int x_end, int y_start, int y_end, int d)
{
    double ret = 0;

    for (int y = y_start; y <= y_end; y++)
    {
        for (int x = x_start; x <= x_end; x++)
        {
            ret += (double)left.at<uchar>(y,x) * (double)right.at<uchar>(y,x-d);
        }
    }

    return ret;
}


void SpeckleStereoMatcher::ZNCC_calc(const cv::Mat &left, const cv::Mat &right, float *cost_volume)
{
    // use integral image to accelerate the calculation
    // https://blog.csdn.net/VisualMan_whu/article/details/38563857
    cv::Mat integral_sum_left, integral_sqsum_left, integral_sum_right, integral_sqsum_right;
    cv::integral(left, integral_sum_left, integral_sqsum_left, CV_64F, CV_64F);
    cv::integral(right, integral_sum_right, integral_sqsum_right, CV_64F, CV_64F);

    for  (int  y = 0; y < _height; y++)
    {
        for (int x = 0; x < _width; x++)
        {
            for (int d =0; d < _max_disparity; d++)
            {
                int x_start = std::max(x - _kernel_size/2, 0);
                int x_end = std::min(x + _kernel_size/2, _width - 1);

                int y_start = std::max(y - _kernel_size/2, 0);
                int y_end = std::min(y + _kernel_size/2, _height - 1);

                if (x_start - d < 0)
                {
                    cost_volume[d + (x + y * _width) * _max_disparity] = 0;
                    continue;
                }

#if 0
                cv::Mat left_roi, right_roi;
                left(cv::Rect(cv::Point(x_start, y_start), cv::Point(x_end, y_end))).convertTo(left_roi, CV_32FC1, 1.0);
                right(cv::Rect(cv::Point(x_start - d, y_start), cv::Point(x_end - d, y_end))).convertTo(right_roi, CV_32FC1, 1.0);

                float left_average = cv::sum(left_roi)[0] / (left_roi.cols * left_roi.rows);
                float right_average = cv::sum(right_roi)[0] / (right_roi.cols * right_roi.rows);

                float xcorr = cv::sum((left_roi - left_average).mul(right_roi - right_average))[0];
                float left_corr = cv::sum((left_roi - left_average).mul(left_roi - left_average))[0];
                float right_corr = cv::sum((right_roi - right_average).mul(right_roi - right_average))[0];
                
                float zncc = xcorr / std::sqrt(left_corr * right_corr);
#else
                double SIJ = calc_xcorr(left, right, x_start, x_end, y_start, y_end, d);
                double SI   = integral_sum_left.at<double>(y_end + 1, x_end + 1)
                            + integral_sum_left.at<double>(y_start, x_start)
                            - integral_sum_left.at<double>(y_start, x_end + 1)
                            - integral_sum_left.at<double>(y_end + 1, x_start);

                double SJ   = integral_sum_right.at<double>(y_end + 1, x_end - d + 1) 
                            + integral_sum_right.at<double>(y_start, x_start - d)
                            - integral_sum_right.at<double>(y_start, x_end - d + 1) 
                            - integral_sum_right.at<double>(y_end + 1, x_start - d);

                double SII  = integral_sqsum_left.at<double>(y_end + 1, x_end + 1) 
                            + integral_sqsum_left.at<double>(y_start, x_start)
                            - integral_sqsum_left.at<double>(y_start, x_end + 1) 
                            - integral_sqsum_left.at<double>(y_end + 1, x_start);

                double SJJ  = integral_sqsum_right.at<double>(y_end + 1, x_end - d + 1) 
                            + integral_sqsum_right.at<double>(y_start, x_start - d)
                            - integral_sqsum_right.at<double>(y_start, x_end - d + 1) 
                            - integral_sqsum_right.at<double>(y_end + 1, x_start - d);

                int N = (x_end - x_start + 1) * (y_end - y_start + 1);
                float left_xcorr = N * SII - SI * SI;
                float right_xcorr = N * SJJ - SJ * SJ;
                float zncc = 0;
                if (left_xcorr != 0 && right_xcorr != 0)
                    zncc = (N * SIJ - SI * SJ) / (std::sqrt(left_xcorr * right_xcorr));
#endif

                cost_volume[d + (x + y * _width) * _max_disparity] = zncc;
                //std::cout << x << " " << y << " " << d << " " << zncc << std::endl;
            }
        }
    }
}

SpeckleStereoMatcher::SpeckleStereoMatcher(int kernel_size, int width, int height):
    _kernel_size(kernel_size), _width(width), _height(height)
{
    _cost_volumn = (float *)malloc(_width * _height * _max_disparity * sizeof(float));
    _cost_volumn_r = (float *)malloc(_width * _height * _max_disparity * sizeof(float));
    _disparity_int = (int *)malloc(_width * _height * sizeof(int));
    _disparity_int_r = (int *)malloc(_width * _height * sizeof(int));
    _disparity_float = (float *)malloc(_width * _height * sizeof(float));

    if (_cost_volumn == NULL || _cost_volumn_r == NULL || _disparity_int == NULL || _disparity_int_r == NULL)
    {
        std::cout << "ERROR: Unable to allocate memory" << std::endl;
    }

    memset(_cost_volumn, 0, _width * _height * _max_disparity * sizeof(float));
    memset(_cost_volumn_r, 0, _width * _height * _max_disparity * sizeof(float));
    memset(_disparity_int, 0, _width * _height * sizeof(int));
    memset(_disparity_int_r, 0, _width * _height * sizeof(int));
    memset(_disparity_float, 0, _width * _height * sizeof(float));
}

SpeckleStereoMatcher::~SpeckleStereoMatcher()
{
    if (_cost_volumn != NULL)
        free(_cost_volumn);

    if (_cost_volumn_r != NULL)
        free(_cost_volumn_r);

    if (_disparity_int != NULL)
        free(_disparity_int);

    if (_disparity_int_r != NULL)
        free(_disparity_int_r);

    if (_disparity_float != NULL)
        free(_disparity_float);
}


void SpeckleStereoMatcher::match(const cv::Mat& left, const cv::Mat& right, cv::Mat& result)
{
#if 0
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, _max_disparity, _kernel_size, 8 * _kernel_size * _kernel_size, 32 * _kernel_size * _kernel_size, 1, 63, 10, 100, 1, cv::StereoSGBM::MODE_HH);
    cv::Mat disp_s16;
    sgbm->compute(left, right, disp_s16);

    disp_s16.convertTo(result, CV_32FC1, 1.0 / 16);
#else
    // 1. calculate the cost volume
    ZNCC_calc(left, right, _cost_volumn);
    std::cout << "ZNCC calculation done" << std::endl;

    // 2. cost agrregation

    // 3. wta
    WTA(_cost_volumn, _disparity_int, _width, _height, _max_disparity);
    std::cout << "WTA" << std::endl;
#if 1
    // 4. LR check
    LR_check(_cost_volumn, _disparity_int, _cost_volumn_r, _disparity_int_r, _width, _height, _max_disparity);
    std::cout << "LR_check" << std::endl;
    
    // 5. refine
    refine(_cost_volumn, _disparity_int, _disparity_float, _width, _height, _max_disparity);
    std::cout << "refine" << std::endl;

    // 6. median filter
    result = cv::Mat::zeros(cv::Size(_width, _height), CV_32FC1);
    median_filter(_disparity_float, (float *)result.data, _width, _height, 5);
    std::cout << "medianBlur" << std::endl;
#else
    result = cv::Mat::zeros(cv::Size(_width, _height), CV_32FC1);
    for (int y = 0; y < _height; y++)
    {
        for (int x = 0; x < _width; x++)
        {
            result.at<float>(y, x) = _disparity_int[x + y * _width];
        }
    }
#endif

#endif
}
