
// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "relativeRotations.hpp"


#include <aliceVision/types.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/robustEstimation/IRansacKernel.hpp>
#include <aliceVision/robustEstimation/conditioning.hpp>

#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/relativePose/Homography4PSolver.hpp>
#include <aliceVision/multiview/relativePose/HomographyError.hpp>
#include <aliceVision/multiview/relativePose/EssentialKernel.hpp>
#include <aliceVision/multiview/relativePose/Rotation3PSolver.hpp>

namespace aliceVision {
namespace sfm {

/**
 * @brief Decompose a homography given known calibration matrices, assuming a pure rotation between the two views.
 * It is supposed that \f$ x_2 \sim H x_1 \f$ with \f$ H = K_2 * R * K_1^{-1} \f$
 * @param[in] homography  3x3 homography matrix H.
 * @return The 3x3 rotation matrix corresponding to the pure rotation between the views.
 */
aliceVision::Mat3 decomposePureRotationHomography(const Mat3 &homography)
{
    // compute the scale factor lambda that makes det(lambda*G) = 1
    const auto lambda = std::pow(1 / homography.determinant(), 1 / 3);
    const auto rotation = lambda * homography;

    //@fixme find possible bad cases?

    // compute and return the closest rotation matrix
    Eigen::JacobiSVD<Mat3> usv(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &u = usv.matrixU();
    const auto vt = usv.matrixV().transpose();

    return u * vt;
}


/**
 * @brief Estimate the homography between two views using corresponding points such that \f$ x_2 \sim H x_1 \f$
 * @param[in] x1 The points on the first image.
 * @param[in] x2 The corresponding points on the second image.
 * @param[in] imgSize1 The size of the first image.
 * @param[in] imgSize2 The size of the second image.
 * @param[out] H The estimated homography.
 * @param[out] vec_inliers The inliers satisfying the homography as a list of indices.
 * @return the status of the estimation.
 */
aliceVision::EstimationStatus robustHomographyEstimationAC(const Mat2X &x1,
                                                           const Mat2X &x2,
                                                           const std::pair<std::size_t, std::size_t> &imgSize1,
                                                           const std::pair<std::size_t, std::size_t> &imgSize2,
                                                           std::mt19937 &randomNumberGenerator,
                                                           Mat3 &H, std::vector<std::size_t> &vec_inliers)
{
    using KernelType = multiview::RelativePoseKernel<multiview::relativePose::Homography4PSolver, multiview::relativePose::HomographyAsymmetricError, multiview::UnnormalizerI, robustEstimation::Mat3Model>;

    // configure as point to point error model.
    KernelType kernel(x1, imgSize1.first, imgSize1.second, x2, imgSize2.first, imgSize2.second, false); 
    
    robustEstimation::Mat3Model model;

    robustEstimation::ACRANSAC(kernel, randomNumberGenerator, vec_inliers, 1024, &model, std::numeric_limits<double>::infinity());
    H = model.getMatrix();

    const bool valid{!vec_inliers.empty()};
    const bool hasStrongSupport{vec_inliers.size() > kernel.getMinimumNbRequiredSamples() * 2.5};

    return {valid, hasStrongSupport};
}

/**
 * @brief Estimate the rotation between two views using corresponding points such that \f$ x_2 = R x_1 \f$
 * @param[in] x1 The points on the first image.
 * @param[in] x2 The corresponding points on the second image.
 * @param[in] imgSize1 The size of the first image.
 * @param[in] imgSize2 The size of the second image.
 * @param[out] R The estimated rotttion.
 * @param[out] vec_inliers The inliers satisfying the rotation as a list of indices.
 * @return the status of the estimation.
 */
aliceVision::EstimationStatus robustRotationEstimationAC(const Mat &x1, const Mat &x2, const std::pair<std::size_t, std::size_t> &imgSize1, const std::pair<std::size_t, std::size_t> &imgSize2, std::mt19937 &randomNumberGenerator,  Mat3 &R, std::vector<std::size_t> &vec_inliers)
{
    using KernelType = multiview::RelativePoseSphericalKernel<multiview::relativePose::Rotation3PSolver, multiview::relativePose::RotationError, robustEstimation::Mat3Model>;

    KernelType kernel(x1, x2);

    robustEstimation::Mat3Model model;
    robustEstimation::ACRANSAC(kernel, randomNumberGenerator, vec_inliers, 1024, &model, std::numeric_limits<double>::infinity());
    R = model.getMatrix();

    const bool valid{!vec_inliers.empty()};

    const bool hasStrongSupport{vec_inliers.size() > kernel.getMinimumNbRequiredSamples() * 2.5}; 

    return {valid, hasStrongSupport};
}

bool robustRelativeRotation_fromE(const Mat3 & K1, const Mat3 & K2, const Mat & x1, const Mat & x2, const std::pair<size_t, size_t> & size_ima1, const std::pair<size_t, size_t> & size_ima2, std::mt19937 &randomNumberGenerator, RelativePoseInfo & relativePose_info, const size_t max_iteration_count)
{
  // Use the 5 point solver to estimate E
  // Define the AContrario adaptor
  using KernelType = multiview::RelativePoseKernel_K<multiview::relativePose::Essential5PSolver, multiview::relativePose::FundamentalSymmetricEpipolarDistanceError, robustEstimation::Mat3Model>;

  KernelType kernel(x1, size_ima1.first, size_ima1.second, x2, size_ima2.first, size_ima2.second, K1, K2);

  // Robustly estimation of the Essential matrix and its precision
  robustEstimation::Mat3Model model;
  const std::pair<double, double> acRansacOut = robustEstimation::ACRANSAC(kernel, randomNumberGenerator, relativePose_info.vec_inliers, max_iteration_count, &model, relativePose_info.initial_residual_tolerance);
  
  relativePose_info.essential_matrix = model.getMatrix();
  relativePose_info.found_residual_precision = acRansacOut.first;

  if (relativePose_info.vec_inliers.size() < kernel.getMinimumNbRequiredSamples() * 2.5); 
  {
    ALICEVISION_LOG_INFO("robustRelativePose: no sufficient coverage (the model does not support enough samples): " << relativePose_info.vec_inliers.size());
    return false; // no sufficient coverage (the model does not support enough samples)
  }

  // estimation of the relative poses
  Mat3 R;
  Vec3 t;
  if (!estimate_Rt_fromE(K1, K2, x1, x2, relativePose_info.essential_matrix, relativePose_info.vec_inliers, &R, &t))
  {
    ALICEVISION_LOG_INFO("robustRelativePose: cannot find a valid [R|t] couple that makes the inliers in front of the camera.");
    return false; // cannot find a valid [R|t] couple that makes the inliers in front of the camera.
  }

  t = Vec3(0.0, 0.0, 0.0);

  // Store [R|C] for the second camera, since the first camera is [Id|0]
  relativePose_info.relativePose = geometry::Pose3(R, -R.transpose() * t);

  return true;
}


bool robustRelativeRotation_fromH(const Mat2X &x1, const Mat2X &x2, const std::pair<size_t, size_t> &imgSize1, const std::pair<size_t, size_t> &imgSize2, std::mt19937 &randomNumberGenerator, RelativeRotationInfo &relativeRotationInfo, const size_t max_iteration_count)
{
  std::vector<std::size_t> vec_inliers{};

  // estimate the homography
  const auto status = robustHomographyEstimationAC(x1, x2, imgSize1, imgSize2, randomNumberGenerator, relativeRotationInfo._homography, relativeRotationInfo._inliers);
  
  if (!status.isValid && !status.hasStrongSupport) {
    return false;
  }

  relativeRotationInfo._relativeRotation = decomposePureRotationHomography(relativeRotationInfo._homography);

  return true;
}

bool robustRelativeRotation_fromR(const Mat &x1, const Mat &x2, const std::pair<size_t, size_t> &imgSize1, const std::pair<size_t, size_t> &imgSize2, std::mt19937 &randomNumberGenerator, RelativeRotationInfo &relativeRotationInfo, const size_t max_iteration_count)
{
  std::vector<std::size_t> vec_inliers{};

  const auto status = robustRotationEstimationAC(x1, x2, imgSize1, imgSize2, randomNumberGenerator, relativeRotationInfo._relativeRotation, relativeRotationInfo._inliers);


  if (!status.isValid && !status.hasStrongSupport) {
    return false;
  }

  return true;
}    

} // namespace sfm
} // namespace aliceVision