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
    inline int find_min_in_array(const float *arr, int len)
    {
        int min_idx = 0;
        float min = arr[0];
        for (int i = 1; i < len; i++)
        {
            if (arr[i] < min)
            {
                min = arr[i];
                min_idx = i;
            }
        }

        return min_idx;
    }

    void WTA(const float *cost_map, int *disparity,
             int width, int height, int max_disparity)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                disparity[x + y * width] = find_min_in_array(cost_map + (x + y * width) * max_disparity, max_disparity);
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

    void cost_aggregation_left2right(const float *cost_map, const uchar *img,
                                     int width, int height, int max_disparity,
                                     float *cost_aggregated, float P1, float P2)
    {
        std::vector<float> cost_last(max_disparity);

        for (int y = 0; y < height; y++)
        {
            // 第一列不需聚合
            memcpy(cost_aggregated, cost_map, max_disparity * sizeof(float));
            memcpy(&cost_last[0], cost_aggregated, max_disparity * sizeof(float));

            float cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

            uchar gray_last = (*img);

            cost_map += max_disparity;
            cost_aggregated += max_disparity;
            img += 1;

            for (int x = 1; x < width; x++)
            {
                uchar gray = (*img);
                float cost_min_cur = UINT8_MAX;
                for (int d = 0; d < max_disparity; d++)
                {
                    float l0 = cost_map[d];
                    float l1 = cost_last[d];
                    float l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                    float l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                    float l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                    cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                    cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
                }

                memcpy(&cost_last[0], cost_aggregated, max_disparity * sizeof(float));
                cost_min_last = cost_min_cur;

                gray_last = gray;

                cost_map += max_disparity;
                cost_aggregated += max_disparity;
                img += 1;
            }
        }
    }

    void cost_aggregation_right2left(const float *cost_map, const uchar *img,
                                     int width, int height, int max_disparity,
                                     float *cost_aggregated, float P1, float P2)
    {
        std::vector<float> cost_last(max_disparity);

        cost_map += (width * height - 1) * max_disparity;
        cost_aggregated += (width * height - 1) * max_disparity;
        img += (width * height - 1);

        for (int y = height - 1; y >= 0; y--)
        {
            // 第一列不需聚合
            memcpy(cost_aggregated, cost_map, max_disparity * sizeof(float));
            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));

            float cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

            uchar gray_last = (*img);

            cost_map -= max_disparity;
            cost_aggregated -= max_disparity;
            img -= 1;

            for (int x = width - 2; x >= 0; x--)
            {
                uchar gray = (*img);
                float cost_min_cur = UINT8_MAX;
                for (int d = 0; d < max_disparity; d++)
                {
                    float l0 = cost_map[d];
                    float l1 = cost_last[d];
                    float l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                    float l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                    float l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                    cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                    cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
                }

                memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));
                cost_min_last = cost_min_cur;

                gray_last = gray;

                cost_map -= max_disparity;
                cost_aggregated -= max_disparity;
                img -= 1;
            }
        }
    }

    void cost_aggregation_up2down(const float *cost_map, const uchar *img,
                                  int width, int height, int max_disparity,
                                  float *cost_aggregated, float P1, float P2)
    {
        std::vector<float> cost_last(max_disparity);

        const float *cost_org = cost_map;
        float *cost_aggre_org = cost_aggregated;
        const uchar *img_data_org = img;

        for (int x = 0; x < width; x++)
        {
            // 赋值
            cost_map = &cost_org[x * max_disparity];
            cost_aggregated = &cost_aggre_org[x * max_disparity];
            img = &img_data_org[x];

            // 第一行不需聚合
            memcpy(cost_aggregated, cost_map, max_disparity * sizeof(float));
            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));

            float cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

            uchar gray_last = (*img);

            cost_map += width * max_disparity;
            cost_aggregated += width * max_disparity;
            img += width;

            for (int y = 1; y < height; y++)
            {
                uchar gray = (*img);
                float cost_min_cur = UINT8_MAX;
                for (int d = 0; d < max_disparity; d++)
                {
                    float l0 = cost_map[d];
                    float l1 = cost_last[d];
                    float l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                    float l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                    float l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                    cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                    cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
                }

                memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));
                cost_min_last = cost_min_cur;

                gray_last = gray;

                cost_map += width * max_disparity;
                cost_aggregated += width * max_disparity;
                img += width;
            }
        }
    }

    void cost_aggregation_down2up(const float *cost_map, const uchar *img,
                                  int width, int height, int max_disparity,
                                  float *cost_aggregated, float P1, float P2)
    {
        std::vector<float> cost_last(max_disparity);

        const float *cost_org = cost_map;
        float *cost_aggre_org = cost_aggregated;
        const uchar *img_data_org = img;

        for (int x = width - 1; x >= 0; x--)
        {
            // 赋值
            cost_map = &cost_org[((height - 1) * width + x) * max_disparity];
            cost_aggregated = &cost_aggre_org[((height - 1) * width + x) * max_disparity];
            img = &img_data_org[((height - 1) * width + x)];

            // 第一行不需聚合
            memcpy(cost_aggregated, cost_map, max_disparity * sizeof(float));
            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));

            float cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

            uchar gray_last = (*img);

            cost_map -= width * max_disparity;
            cost_aggregated -= width * max_disparity;
            img -= width;

            for (int y = height - 2; y >= 0; y--)
            {
                uchar gray = (*img);
                float cost_min_cur = UINT8_MAX;
                for (int d = 0; d < max_disparity; d++)
                {
                    float l0 = cost_map[d];
                    float l1 = cost_last[d];
                    float l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                    float l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                    float l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                    cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                    cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
                }

                memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(float));
                cost_min_last = cost_min_cur;

                gray_last = gray;

                cost_map -= width * max_disparity;
                cost_aggregated -= width * max_disparity;
                img -= width;
            }
        }
    }

    void cost_aggregation(const float *cost_map, const uchar *img,
                          int width, int height, int max_disparity,
                          float *cost_aggregated, float P1, float P2,
                          float *cost_scanline_buffer)
    {
        const int scanline_path = 4;
        float *cost_scanline[scanline_path];

        for (int i = 0; i < scanline_path; i++)
        {
            cost_scanline[i] = cost_scanline_buffer + i * width * height * max_disparity;
        }

        cost_aggregation_left2right(cost_map, img, width, height, max_disparity, cost_scanline[0], P1, P2);
        cost_aggregation_right2left(cost_map, img, width, height, max_disparity, cost_scanline[1], P1, P2);
        cost_aggregation_up2down(cost_map, img, width, height, max_disparity, cost_scanline[2], P1, P2);
        cost_aggregation_down2up(cost_map, img, width, height, max_disparity, cost_scanline[3], P1, P2);

        for (int i = 0; i < width * height * max_disparity; i++)
        {
            float cost_i = 0;

            cost_i += cost_scanline[0][i] +
                      cost_scanline[1][i] +
                      cost_scanline[2][i] +
                      cost_scanline[3][i];

            cost_aggregated[i] = (cost_i / scanline_path);
        }
    }
}

#endif // __STEREO_MATCH_HELPER_H__
