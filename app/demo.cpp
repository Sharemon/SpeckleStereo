/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-08
 */

#include <iostream>
#include <fstream>
#include "../src/SpeckleStereoMatcher.hpp"

void convert_disparity_map_to_point_cloud(const cv::Mat &disp, std::vector<cv::Vec6f> &point_cloud_with_texture, const cv::Mat &Q, const cv::Mat &texture)
{
    point_cloud_with_texture.clear();

    int width = disp.cols;
    int height = disp.rows;

    double cx = -Q.at<double>(0, 3);
    double cy = -Q.at<double>(1, 3);
    double f = Q.at<double>(2, 3);
    double w = Q.at<double>(3, 2);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float d = disp.at<float>(y, x);
            if (d < 20) // set a distance max limit of 3m
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
                uchar grayscale = texture.at<uchar>(y, x);
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

            point_cloud_with_texture.push_back(xyz_rgb);
        }
    }
}

void test_speckle_match_for_single_image(int argc, char const *argv[])
{
    // 1. parse arguments
    if (argc < 4)
    {
        std::cout << "Usage: ./SpeckleStereo [image path] [stereo parameters path] [output path]" << std::endl;
        return;
    }

    std::string img_path = argv[1];
    std::string param_path = argv[2];
    std::string output_path = argv[3];

    // 2. read image
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat left = img.colRange(0, img.cols / 2);
    cv::Mat right = img.colRange(img.cols / 2, img.cols);

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
    std::cout << "Kl:\n"
              << Kl << std::endl;
    std::cout << "Dl:\n"
              << Dl << std::endl;
    std::cout << "Kr:\n"
              << Kr << std::endl;
    std::cout << "Dr:\n"
              << Dr << std::endl;
    std::cout << "R: \n"
              << R << std::endl;
    std::cout << "T: \n"
              << T << std::endl;

    // 4. stereo rectify
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(Kl, Dl, Kr, Dr, left.size(), R, T, R1, R2, P1, P2, Q);
    std::cout << "============== rectify parameters ==================" << std::endl;
    std::cout << "P1:\n"
              << P1 << std::endl;
    std::cout << "P2:\n"
              << P2 << std::endl;
    std::cout << "Q: \n"
              << Q << std::endl;

    cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
    cv::initUndistortRectifyMap(Kl, Dl, R1, P1, left.size(), CV_32FC1, left_mapx, left_mapy);
    cv::initUndistortRectifyMap(Kr, Dr, R2, P2, right.size(), CV_32FC1, right_mapx, right_mapy);

    cv::Mat left_rectified, right_rectified;
    cv::remap(left, left_rectified, left_mapx, left_mapy, cv::INTER_LINEAR);
    cv::remap(right, right_rectified, right_mapx, right_mapy, cv::INTER_LINEAR);

    cv::imwrite("left.png", left_rectified);
    cv::imwrite("right.png", right_rectified);

    cv::Mat merge;
    cv::hconcat(left_rectified, right_rectified, merge);
    cv::imwrite("merge.png", merge);

    cv::resize(left_rectified, left_rectified, left_rectified.size());
    cv::resize(right_rectified, right_rectified, right_rectified.size());

    // 5. stereo match
    SpeckleStereo::SpeckleStereoMatcher matcher(11, left_rectified.cols, left_rectified.rows);

    cv::Mat disp;
    auto t0 = std::chrono::high_resolution_clock::now();
    matcher.match(left_rectified, right_rectified, disp);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

    // 6. save result
    if (!disp.empty())
    {
        // 6.1. save disparity map
        FILE *fp = fopen((output_path + "/disparity_map.raw").c_str(), "wb");
        fwrite(disp.data, disp.cols * disp.rows * sizeof(float), 1, fp);
        fclose(fp);

        cv::Mat disp_u8;
        disp.convertTo(disp_u8, CV_8UC1, 1.0);
        cv::imwrite("disp.png", disp_u8);

        // 6.2 save point cloud
        std::vector<cv::Vec6f> point_cloud_with_texture;
        convert_disparity_map_to_point_cloud(disp, point_cloud_with_texture, Q, left_rectified);

        std::ofstream point_cloud_file(output_path + "/point_cloud.txt");
        for (auto point : point_cloud_with_texture)
        {
            point_cloud_file << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << " " << point[4] << " " << point[5] << std::endl;
        }
        point_cloud_file.close();
    }
}

void test_speckle_match_for_triple_image(int argc, char const *argv[])
{
    // 1. parse arguments
    if (argc < 6)
    {
        std::cout << "Usage: ./SpeckleStereo [image1 path] [image2 path] [image3 path] [stereo parameters path] [output path]" << std::endl;
        return;
    }

    std::string img_path[3] = {std::string(argv[1]), std::string(argv[2]), std::string(argv[3])};
    std::string param_path = argv[4];
    std::string output_path = argv[5];

    // 2. read stereo parameters
    cv::FileStorage param_file(param_path, cv::FileStorage::READ);

    cv::Mat Kl, Dl, Kr, Dr, R, T;
    param_file["Kl"] >> Kl;
    param_file["Dl"] >> Dl;
    param_file["Kr"] >> Kr;
    param_file["Dr"] >> Dr;
    param_file["R"] >> R;
    param_file["T"] >> T;

    std::cout << "============== read parameters ==================" << std::endl;
    std::cout << "Kl:\n"
              << Kl << std::endl;
    std::cout << "Dl:\n"
              << Dl << std::endl;
    std::cout << "Kr:\n"
              << Kr << std::endl;
    std::cout << "Dr:\n"
              << Dr << std::endl;
    std::cout << "R: \n"
              << R << std::endl;
    std::cout << "T: \n"
              << T << std::endl;

    std::vector<cv::Mat> left_rectified_images, right_rectified_images;
    cv::Mat R1, R2, P1, P2, Q;
    for (int i = 0; i < 3; i++)
    {
        // 3. read image
        cv::Mat img = cv::imread(img_path[i], cv::IMREAD_GRAYSCALE);
        cv::Mat left = img.colRange(0, img.cols / 2);
        cv::Mat right = img.colRange(img.cols / 2, img.cols);

        // 4. stereo rectify
        cv::stereoRectify(Kl, Dl, Kr, Dr, left.size(), R, T, R1, R2, P1, P2, Q);
        std::cout << "============== rectify parameters ==================" << std::endl;
        std::cout << "P1:\n"
                  << P1 << std::endl;
        std::cout << "P2:\n"
                  << P2 << std::endl;
        std::cout << "Q: \n"
                  << Q << std::endl;

        cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
        cv::initUndistortRectifyMap(Kl, Dl, R1, P1, left.size(), CV_32FC1, left_mapx, left_mapy);
        cv::initUndistortRectifyMap(Kr, Dr, R2, P2, right.size(), CV_32FC1, right_mapx, right_mapy);

        cv::Mat left_rectified, right_rectified;
        cv::remap(left, left_rectified, left_mapx, left_mapy, cv::INTER_LINEAR);
        cv::remap(right, right_rectified, right_mapx, right_mapy, cv::INTER_LINEAR);

        cv::imwrite("left" + std::to_string(i) + ".png", left_rectified);
        cv::imwrite("right" + std::to_string(i) + ".png", right_rectified);

        cv::Mat merge;
        cv::hconcat(left_rectified, right_rectified, merge);
        cv::imwrite("merge" + std::to_string(i) + ".png", merge);

        left_rectified_images.push_back(left_rectified);
        right_rectified_images.push_back(right_rectified);
    }

    // 5. stereo match
    SpeckleStereo::SpeckleStereoMatcher matcher(11, left_rectified_images[0].cols, right_rectified_images[0].rows);

    cv::Mat disp;
    auto t0 = std::chrono::high_resolution_clock::now();
    matcher.match(left_rectified_images, right_rectified_images, disp);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

    // 6. save result
    if (!disp.empty())
    {
        // 6.1. save disparity map
        FILE *fp = fopen((output_path + "/disparity_map.raw").c_str(), "wb");
        fwrite(disp.data, disp.cols * disp.rows * sizeof(float), 1, fp);
        fclose(fp);

        cv::Mat disp_u8;
        disp.convertTo(disp_u8, CV_8UC1, 1.0);
        cv::imwrite("disp.png", disp_u8);

        // 6.2 save point cloud
        std::vector<cv::Vec6f> point_cloud_with_texture;
        convert_disparity_map_to_point_cloud(disp, point_cloud_with_texture, Q, left_rectified_images[0]);

        std::ofstream point_cloud_file(output_path + "/point_cloud.txt");
        for (auto point : point_cloud_with_texture)
        {
            point_cloud_file << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << " " << point[4] << " " << point[5] << std::endl;
        }
        point_cloud_file.close();
    }
}

int main(int argc, char const *argv[])
{
#if 0
    test_speckle_match_for_single_image(argc, argv);
#else
    test_speckle_match_for_triple_image(argc, argv);
#endif

    return 0;
}