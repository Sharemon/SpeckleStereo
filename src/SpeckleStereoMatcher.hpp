/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-08
 */

#if !defined(__SPECKLE_STEREO_MATCHER_H__)
#define __SPECKLE_STEREO_MATCHER_H__

#include <opencv2/opencv.hpp>

namespace SpeckleStereo
{
    class SpeckleStereoMatcher
    {
    private:
        const int _max_disparity =160;
        int _kernel_size;
        int _width;
        int _height;

        float *_cost_volumn = NULL;
        float *_cost_volumn_aggr = NULL;
        float *_scaneline_buffer = NULL;
        float *_cost_volumn_r = NULL;
        int *_disparity_int = NULL;
        int *_disparity_int_r = NULL;
        float *_disparity_float = NULL;

        void ZNCC_calc(const cv::Mat &left, const cv::Mat &right, float *cost_volume);
        void ZNCC_calc_tripple_images(const std::vector<cv::Mat> &lefts, const std::vector<cv::Mat> &rights, float *cost_volume);

    public:
        SpeckleStereoMatcher(int kernel_size, int width, int height);
        ~SpeckleStereoMatcher();

        void match(const cv::Mat& left, const cv::Mat& right, cv::Mat& result);
        void match(const std::vector<cv::Mat>& lefts, const std::vector<cv::Mat>& rights, cv::Mat& result);
    };
   
}


#endif // __SPECKLE_STEREO_MATCHER_H__

