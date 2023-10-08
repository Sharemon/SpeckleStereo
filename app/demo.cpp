/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-08
 */

#include <iostream>
#include <fstream>
#include "../src/SpeckleStereoMatcher.hpp"

void convert_disparity_map_to_point_cloud(const cv::Mat& disp, std::vector<cv::Vec6f> point_cloud_with_texture, const cv::Mat& Q, const cv::Mat& texture)
{
    point_cloud_with_texture.clear();

    int width = disp.cols;
    int height = disp.rows;

    double cx = Q.at<double>(0, 3);
    double cy = Q.at<double>(1, 3);
    double f  = Q.at<double>(2, 3);
    double w  = Q.at<double>(3, 2);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float d = disp.at<float>(y, x);
            if (d == 0)
                continue;

            float dw = d * w;

            float X = (x - cx) / dw;
            float Y = (y - cy) / dw;
            float Z = f / dw;

            cv::Vec6f xyz_rgb;
            xyz_rgb[0] = X;
            xyz_rgb[1] = Y;
            xyz_rgb[2] = Z;

            if (texture.channels() == 1)
            {
                uchar grayscale = texture.at<uchar>(y,x);
                xyz_rgb[3] = grayscale;
                xyz_rgb[4] = grayscale;
                xyz_rgb[5] = grayscale;
            }
            else
            {
                cv::Vec3b rgb = texture.at<cv::Vec3b>(y, x);
                xyz_rgb[3] = rgb[0];
                xyz_rgb[4] = rgb[1];
                xyz_rgb[5] = rgb[2];
            }
        }
    }
}


int main(int argc, char const *argv[])
{
    // 1. parse arguments
    if (argc < 4)
    {
        std::cout << "Usage: ./SpeckleStereo [image path] [stereo parameters path] [output path]" << std::endl;
    }

    std::string img_path = argv[1];
    std::string param_path = argv[2];
    std::string output_path = argv[3];

    // 2. read image
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat left = img.colRange(0, img.cols/2);
    cv::Mat right = img.colRange(img.cols/2, img.cols);

    // 3. read stereo parameters
    cv::FileStorage param_file(param_path, cv::FileStorage::READ);

    cv::Mat Kl, Dl, Kr, Dr, R, T;
    param_file["Kl"] >> Kl;
    param_file["Dl"] >> Dl;
    param_file["Kr"] >> Kr;
    param_file["Dr"] >> Dr;
    param_file["R"] >> R;
    param_file["T"] >> T;

    std::cout << "============== read parameters ==================" << std::endl;
    std::cout << "Kl:\n" << Kl << std::endl;
    std::cout << "Dl:\n" << Dl << std::endl;
    std::cout << "Kr:\n" << Kr << std::endl;
    std::cout << "Dr:\n" << Dr << std::endl;
    std::cout << "R: \n" << R  << std::endl;
    std::cout << "T: \n" << T  << std::endl;

    // 4. stereo rectify
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(Kl, Dl, Kr, Dr, left.size(), R, T, R1, R2, P1, P2, Q);
    std::cout << "============== rectify parameters ==================" << std::endl;
    std::cout << "P1:\n" << P1 << std::endl;
    std::cout << "P2:\n" << P2 << std::endl;
    std::cout << "Q: \n" << Q  << std::endl;

    cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
    cv::initUndistortRectifyMap(Kl, Dl, R1, P1, left.size(), CV_32FC1, left_mapx, left_mapy);
    cv::initUndistortRectifyMap(Kr, Dr, R2, P2, right.size(), CV_32FC1, right_mapx, right_mapy);

    cv::Mat left_rectified, right_rectified;
    cv::remap(left, left_rectified, left_mapx, left_mapy, cv::INTER_LINEAR);
    cv::remap(right, right_rectified, right_mapx, right_mapy, cv::INTER_LINEAR);

    // 5. stereo match
    SpeckleStereo::SpeckleStereoMatcher matcher;

    cv::Mat disp;
    matcher.match(left_rectified, right_rectified, disp);

    // 6. save result
    if (!disp.empty())
    {
        // 6.1. save disparity map
        FILE *fp = fopen((output_path + "/disparity_map.raw").c_str(), "wb");
        fwrite(disp.data, disp.cols * disp.rows * sizeof(float), 1, fp);
        fclose(fp);

        // 6.2 save point cloud
        std::vector<cv::Vec6f> point_cloud_with_texture;
        convert_disparity_map_to_point_cloud(disp, point_cloud_with_texture, Q, left);

        std::ofstream point_cloud_file(output_path + "/point_cloud.txt");
        for (auto point : point_cloud_with_texture)
        {
            point_cloud_file << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << " " << point[4] << " " << point[5] << std::endl;
        }
        point_cloud_file.close();
    }

    return 0;
}