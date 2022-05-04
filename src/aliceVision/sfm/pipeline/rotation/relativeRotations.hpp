// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/sfm/pipeline/RelativePoseInfo.hpp>

#include <vector>
#include <random>

namespace aliceVision {
namespace sfm {

enum ERelativeRotationMethod
{
    RELATIVE_ROTATION_FROM_E = 0,
    RELATIVE_ROTATION_FROM_R = 1,
    RELATIVE_ROTATION_FROM_H = 2
};

inline std::string ERelativeRotationMethod_enumToString(const ERelativeRotationMethod rotationMethod)
{
    switch (rotationMethod)
    {
    case ERelativeRotationMethod::RELATIVE_ROTATION_FROM_E:      return "essential_matrix";
    case ERelativeRotationMethod::RELATIVE_ROTATION_FROM_R:   return "rotation_matrix";
    case ERelativeRotationMethod::RELATIVE_ROTATION_FROM_H:   return "homography_matrix";
    }
    throw std::out_of_range("Invalid method name enum");
}

inline ERelativeRotationMethod ERelativeRotationMethod_stringToEnum(const std::string& rotationMethodName)
{
    std::string methodName = rotationMethodName;
    std::transform(methodName.begin(), methodName.end(), methodName.begin(), ::tolower);

    if (methodName == "essential_matrix") return ERelativeRotationMethod::RELATIVE_ROTATION_FROM_E;
    if (methodName == "rotation_matrix") return ERelativeRotationMethod::RELATIVE_ROTATION_FROM_R;
    if (methodName == "homography_matrix") return ERelativeRotationMethod::RELATIVE_ROTATION_FROM_H;

    throw std::out_of_range("Invalid method name : '" + rotationMethodName + "'");
}

inline std::ostream& operator<<(std::ostream& os, ERelativeRotationMethod rotationMethodName)
{
    os << ERelativeRotationMethod_enumToString(rotationMethodName);
    return os;
}

inline std::istream& operator>>(std::istream& in, ERelativeRotationMethod& rotationMethod)
{
    std::string token;
    in >> token;
    rotationMethod = ERelativeRotationMethod_stringToEnum(token);
    return in;
}

/**
 * @brief A struct containing the information of the relative rotation.
 */
struct RelativeRotationInfo
{
    /**
     * @brief Default constructor.
     */
    RelativeRotationInfo() = default;

    /// the homography.
    Mat3 _homography{};
    /// the relative rotation.
    Mat3 _relativeRotation{};
    /// the inliers.
    std::vector<size_t> _inliers{};
    /// initial threshold for the acransac process.
    double _initialResidualTolerance{std::numeric_limits<double>::infinity()};
    /// the estimated threshold found by acransac.
    double _foundResidualPrecision{std::numeric_limits<double>::infinity()};

};

/**
 * @brief Estimate the relative pose between two views.
 * @param[in] K1 3x3 calibration matrix of the first view.
 * @param[in] K2 3x3 calibration matrix of the second view.
 * @param[in] x1 The points on the first image.
 * @param[in] x2 The corresponding points on the second image.
 * @param[in] imgSize1 The size of the first image.
 * @param[in] imgSize2 The size of the second image.
 * @param[in] randomNumberGenerator random number generator
 * @param[out] relativePoseInfo Contains the result of the estimation.
 * @param[in] maxIterationCount Max number of iteration for the ransac process.
 * @return true if a homography has been estimated.
 */
bool robustRelativeRotation_fromE(const Mat3 & K1, const Mat3 & K2,
                                  const Mat & x1, const Mat & x2,
                                  const std::pair<size_t, size_t> & size_ima1,
                                  const std::pair<size_t, size_t> & size_ima2,
                                  std::mt19937 &randomNumberGenerator,
                                  RelativePoseInfo & relativePose_info,
                                  const size_t max_iteration_count = 4096);

/**
 * @brief Estimate the relative rotation between two views related by a pure rotation.
 * @param[in] x1 The points on the first image.
 * @param[in] x2 The corresponding points on the second image.
 * @param[in] imgSize1 The size of the first image.
 * @param[in] imgSize2 The size of the second image.
 * @param[in] randomNumberGenerator random number generator
 * @param[out] relativeRotationInfo Contains the result of the estimation.
 * @param[in] maxIterationCount Max number of iteration for the ransac process.
 * @return true if a homography has been estimated.
 */
bool robustRelativeRotation_fromH(const Mat2X &x1, const Mat2X &x2,
                                  const std::pair<size_t, size_t> &imgSize1,
                                  const std::pair<size_t, size_t> &imgSize2,
                                  std::mt19937 &randomNumberGenerator,
                                  RelativeRotationInfo &relativeRotationInfo,
                                  const size_t max_iteration_count = 4096);

/**
 * @brief Estimate the relative rotation between two views related by a pure rotation.
 * @param[in] x1 The points on the first image.
 * @param[in] x2 The corresponding points on the second image.
 * @param[in] imgSize1 The size of the first image.
 * @param[in] imgSize2 The size of the second image.
 * @param[in] randomNumberGenerator random number generator
 * @param[out] relativeRotationInfo Contains the result of the estimation.
 * @param[in] maxIterationCount Max number of iteration for the ransac process.
 * @return true if a homography has been estimated.
 */
bool robustRelativeRotation_fromR(const Mat &x1, const Mat &x2,
                                  const std::pair<size_t, size_t> &imgSize1,
                                  const std::pair<size_t, size_t> &imgSize2,
                                  std::mt19937 &randomNumberGenerator,
                                  RelativeRotationInfo &relativeRotationInfo,
                                  const size_t max_iteration_count = 4096);

} // namespace sfm
} // namespace aliceVision

