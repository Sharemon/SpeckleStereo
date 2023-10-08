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
        /* data */
    public:
        SpeckleStereoMatcher(/* args */);
        ~SpeckleStereoMatcher();

        void match(const cv::Mat& left, const cv::Mat& right, cv::Mat& result);
    };
   
}


#endif // __SPECKLE_STEREO_MATCHER_H__

