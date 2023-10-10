/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-09
 */

#if !defined(__STEREO_MATCH_HELPER_H__)
#define __STEREO_MATCH_HELPER_H__

#include <iostream>

namespace SpeckleStereo
{
    inline int find_max_in_array(const float *arr, int len)
    {
        int max_idx = 0;
        float max = arr[0];
        for (int i = 1; i < len; i++)
        {
            if (arr[i] > max)
            {
                max = arr[i];
                max_idx = i;
            }
        }

        return max_idx;
    }


    void WTA(const float *cost_map, int *disparity,
                  int width, int height, int max_disparity)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                disparity[x + y * width] = find_max_in_array(cost_map + (x + y * width) * max_disparity, max_disparity);
            }
        }
    }

    void LR_check(const float *cost_map, int *disparity,
                       float *cost_map_r, int *disparity_r,
                       int width, int height, int max_disparity)
    {

        // 右图代价空间构建
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int d = 0; d < max_disparity; d++)
                {
                    if (x + d < width)
                        cost_map_r[(y * width + x) * max_disparity + d] = cost_map[(y * width + x + d) * max_disparity + d];
                }
            }
        }

        // 求右图视差
        WTA(cost_map_r, disparity_r, width, height, max_disparity);

        // 左右一致性检查
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int l = disparity[x + y * width];
                if (x - l < 0)
                    continue;

                int r = disparity_r[x - l + y * width];
                if (std::abs(l - r) > 1)
                {
                    disparity[x + y * width] = 0;
                }
            }
        }
    }

    void refine(const float *cost_map, const int *disparity, float *disparity_float,
                     int width, int height, int max_disparity)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int d = disparity[x + y * width];
                if (d != 0 && d != max_disparity - 1)
                {
                    float c0 = cost_map[(y * width + x) * max_disparity + d];
                    float c1 = cost_map[(y * width + x) * max_disparity + d - 1];
                    float c2 = cost_map[(y * width + x) * max_disparity + d + 1];

                    float demon = c1 + c2 - 2 * c0;
                    float dsub = d + (c1 - c2) / demon / 2.0f;

                    disparity_float[x + y * width] = dsub;
                }
                else
                {
                    disparity_float[x + y * width] = d;
                }
            }
        }
    }

    void median_filter(float *disparity_in, float *disparity_out, int width, int height, int kernel_size)
    {
        std::vector<float> win;

        for (int y = kernel_size / 2; y < height - kernel_size / 2; y++)
        {
            for (int x = kernel_size / 2; x < width - kernel_size / 2; x++)
            {
                win.clear();

                for (int r = -kernel_size / 2; r <= kernel_size / 2; r++)
                {
                    for (int c = -kernel_size / 2; c <= kernel_size / 2; c++)
                    {
                        win.push_back(disparity_in[x + c + (y + r) * width]);
                    }
                }

                std::sort(win.begin(), win.end());
                disparity_out[x + y * width] = win[win.size() / 2];
            }
        }
    }
}

#endif // __STEREO_MATCH_HELPER_H__
