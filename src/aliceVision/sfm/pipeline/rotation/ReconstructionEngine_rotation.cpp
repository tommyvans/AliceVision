// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ReconstructionEngine_rotation.hpp"
#include <aliceVision/sfmData/SfMData.hpp>
#include "relativeRotations.hpp"

namespace aliceVision
{
namespace sfm
{

ReconstructionEngine_rotation::ReconstructionEngine_rotation(const sfmData::SfMData& sfmData,
                                                             const ReconstructionEngine_rotation::Params& params,
                                                             const std::string& outDirectory)
: ReconstructionEngine(sfmData, outDirectory), _params(params)
{
}

ReconstructionEngine_rotation::~ReconstructionEngine_rotation() {}

void ReconstructionEngine_rotation::SetFeaturesProvider(feature::FeaturesPerView* featuresPerView)
{
    _featuresPerView = featuresPerView;
}

void ReconstructionEngine_rotation::SetMatchesProvider(matching::PairwiseMatches* provider)
{
    _pairwiseMatches = provider;
}

bool ReconstructionEngine_rotation::process()
{
    aliceVision::rotationAveraging::RelativeRotations relatives_R;
    Compute_Relative_Rotations(relatives_R);

    HashMap<IndexT, Mat3> global_rotations;
    if(!Compute_Global_Rotations(relatives_R, global_rotations))
    {
        ALICEVISION_LOG_WARNING("Panorama:: Rotation Averaging failure!");
        return false;
    }

    // we set translation vector to zero
    for(const auto& gR : global_rotations)
    {
        const Vec3 t(0.0, 0.0, 0.0);
        const IndexT poseId = gR.first;
        const Mat3& Ri = gR.second;
        _sfmData.setAbsolutePose(poseId, sfmData::CameraPose(geometry::Pose3(Ri, t)));
    }

    return true;
}

/// Compute from relative rotations the global rotations of the camera poses
bool ReconstructionEngine_rotation::Compute_Global_Rotations(const rotationAveraging::RelativeRotations& relatives_R, HashMap<IndexT, Mat3>& global_rotations)
{
    if(relatives_R.empty())
    {
        return false;
    }

    rotationAveraging::RelativeRotations local_relatives_R = relatives_R;

    // Create set of unique pose pairs
    std::set<IndexT> set_pose_ids;
    for(const auto& relative_R : local_relatives_R)
    {
        set_pose_ids.insert(relative_R.i);
        set_pose_ids.insert(relative_R.j);
    }

    // Global Rotation solver
    const ERelativeRotationInferenceMethod eRelativeRotationInferenceMethod = TRIPLET_ROTATION_INFERENCE_NONE;

    //-- Rejection triplet that are 'not' identity rotation
    GlobalSfMRotationAveragingSolver rotationAveraging_solver;
    const bool b_rotationAveraging = rotationAveraging_solver.Run(_params.eRotationAveragingMethod, eRelativeRotationInferenceMethod, local_relatives_R, _params.maxAngularError, global_rotations);

    ALICEVISION_LOG_DEBUG("Found #global_rotations: " << global_rotations.size());

    return b_rotationAveraging;
}

void ReconstructionEngine_rotation::Compute_Relative_Rotations(rotationAveraging::RelativeRotations& vec_relatives_R)
{
    //
    // Build the Relative pose graph from matches:
    //

    // If there is pose priors in the sfmData, use them
    sfmData::RotationPriors& rotationpriors = _sfmData.getRotationPriors();
    for(auto& iter_v1 : _sfmData.getViews())
    {

        // Make sure we have info for this pose
        if(!_sfmData.isPoseAndIntrinsicDefined(iter_v1.first))
        {
            continue;
        }

        for(auto& iter_v2 : _sfmData.getViews())
        {

            // Only different pairs
            if(iter_v1.first == iter_v2.first)
            {
                continue;
            }

            // Make sure we have info for this pose
            if(!_sfmData.isPoseAndIntrinsicDefined(iter_v2.first))
            {
                continue;
            }

            IndexT pid1 = iter_v1.second->getPoseId();
            IndexT pid2 = iter_v2.second->getPoseId();


            //Compute relative pose between the two views
            sfmData::CameraPose oneTo = _sfmData.getAbsolutePose(iter_v1.second->getPoseId());
            sfmData::CameraPose twoTo = _sfmData.getAbsolutePose(iter_v2.second->getPoseId());
            Eigen::Matrix3d oneRo = oneTo.getTransform().rotation();
            Eigen::Matrix3d twoRo = twoTo.getTransform().rotation();
            Eigen::Matrix3d twoRone = twoRo * oneRo.transpose();

            // Create a rotation prior for further sfm
            sfmData::RotationPrior prior(iter_v1.first, iter_v2.first, twoRone);
            rotationpriors.push_back(prior);

            // Add prior on relative rotations with a low weight
            vec_relatives_R.emplace_back(iter_v1.first, iter_v2.first, twoRone, _params.rotationAveragingWeighting ? 1.0 : 0.01);
        }
    }

    sfmData::Constraints2D & constraints2d = _sfmData.getConstraints2D();
    ALICEVISION_LOG_INFO("Relative pose computation:");

    // For each pair of matching views, compute the relative pose
    for(auto pair : *_pairwiseMatches)
    {
        const Pair view_pair = pair.first;

        // If a pair has the same ID, discard it
        if(view_pair.first == view_pair.second)
        {
            continue;
        }

        const IndexT I = view_pair.first;
        const IndexT J = view_pair.second;
        const sfmData::View* view_I = _sfmData.views[I].get();
        const sfmData::View* view_J = _sfmData.views[J].get();

        // Check that valid cameras are existing for the pair of view
        if(_sfmData.getIntrinsics().count(view_I->getIntrinsicId()) == 0 || _sfmData.getIntrinsics().count(view_J->getIntrinsicId()) == 0)
        {
            continue;
        }

        // Get Cameras
        std::shared_ptr<camera::IntrinsicBase> cam_I = _sfmData.getIntrinsics().at(view_I->getIntrinsicId());
        std::shared_ptr<camera::IntrinsicBase> cam_J = _sfmData.getIntrinsics().at(view_J->getIntrinsicId());
        std::shared_ptr<camera::EquiDistant> cam_I_equidistant = std::dynamic_pointer_cast<camera::EquiDistant>(cam_I);
        std::shared_ptr<camera::EquiDistant> cam_J_equidistant = std::dynamic_pointer_cast<camera::EquiDistant>(cam_J);

        bool useSpherical = false;
        if(cam_I_equidistant && cam_J_equidistant)
        {
            useSpherical = true;
        }

        if(_params.eRelativeRotationMethod == RELATIVE_ROTATION_FROM_R)
        {
            useSpherical = true;
        }

        // Build a list of pairs in meters
        const matching::MatchesPerDescType& matchesPerDesc = pair.second;
        std::size_t nbBearing = matchesPerDesc.getNbAllMatches();

        // Create containers for points
        std::size_t iBearing = 0;
        Mat x1, x2;
        if(useSpherical)
        {
            x1 = Mat(3, nbBearing);
            x2 = Mat(3, nbBearing);
        }
        else
        {
            x1 = Mat(2, nbBearing);
            x2 = Mat(2, nbBearing);
        }

        
        //For each matched feature of this pair
        for(const auto& matchesPerDescIt : matchesPerDesc)
        {
            const feature::EImageDescriberType descType = matchesPerDescIt.first;
            assert(descType != feature::EImageDescriberType::UNINITIALIZED);
            const matching::IndMatches& matches = matchesPerDescIt.second;

            const feature::PointFeatures& feats_I = _featuresPerView->getFeatures(I, descType);
            const feature::PointFeatures& feats_J = _featuresPerView->getFeatures(J, descType);

            for(const auto& match : matches)
            {
                const feature::PointFeature & feat_I = feats_I[match._i];
                const feature::PointFeature & feat_J = feats_J[match._j];

                const Vec3 bearingVector_I = cam_I->toUnitSphere(cam_I->removeDistortion(cam_I->ima2cam(feat_I.coords().cast<double>())));
                const Vec3 bearingVector_J = cam_J->toUnitSphere(cam_J->removeDistortion(cam_J->ima2cam(feat_J.coords().cast<double>())));

                if(useSpherical)
                {
                    x1.col(iBearing) = bearingVector_I;
                    x2.col(iBearing) = bearingVector_J;
                }
                else
                {
                    x1.col(iBearing) = bearingVector_I.head(2) / bearingVector_I(2);
                    x2.col(iBearing) = bearingVector_J.head(2) / bearingVector_J(2);
                }

                iBearing ++;
            }
        }
        assert(nbBearing == iBearing);

        
        // Compute max authorized error as geometric mean of camera plane tolerated residual error
        RelativePoseInfo relativePose_info;
        relativePose_info.initial_residual_tolerance = std::pow(cam_I->imagePlaneToCameraPlaneError(2.5) * cam_J->imagePlaneToCameraPlaneError(2.5), 1. / 2.);

        // Since we use normalized features, we will use unit image size and intrinsic matrix:
        const std::pair<size_t, size_t> imageSize(1., 1.);
        const Mat3 K = Mat3::Identity();

        switch(_params.eRelativeRotationMethod)
        {
            //If we use Essential matrix
            case RELATIVE_ROTATION_FROM_E:
            {
                if(!robustRelativeRotation_fromE(K, K, x1, x2, imageSize, imageSize, _randomNumberGenerator, relativePose_info))
                {
                    ALICEVISION_LOG_INFO("Relative pose computation: (" << I << ", " << J << ") => FAILED");
                    continue;
                }
            }
            break;

            //If we use Homography matrix
            case RELATIVE_ROTATION_FROM_H:
            {
                RelativeRotationInfo relativeRotation_info;
                relativeRotation_info._initialResidualTolerance = std::pow(cam_I->imagePlaneToCameraPlaneError(2.5) * cam_J->imagePlaneToCameraPlaneError(2.5), 1. / 2.);

                if(!robustRelativeRotation_fromH(x1, x2, imageSize, imageSize, _randomNumberGenerator, relativeRotation_info))
                {
                    ALICEVISION_LOG_INFO("Relative pose computation: (" << I << ", " << J << ") => FAILED");
                    continue;
                }

                relativePose_info.relativePose = geometry::Pose3(relativeRotation_info._relativeRotation, Vec3::Zero());
                relativePose_info.initial_residual_tolerance = relativeRotation_info._initialResidualTolerance;
                relativePose_info.found_residual_precision = relativeRotation_info._foundResidualPrecision;
                relativePose_info.vec_inliers = relativeRotation_info._inliers;
            }
            break;

            //If we use Rotation matrix
            case RELATIVE_ROTATION_FROM_R:
            {
                RelativeRotationInfo relativeRotation_info;
                relativeRotation_info._initialResidualTolerance = std::pow(cam_I->imagePlaneToCameraPlaneError(2.5) * cam_J->imagePlaneToCameraPlaneError(2.5), 1. / 2.);

                if(!robustRelativeRotation_fromR(x1, x2, imageSize, imageSize, _randomNumberGenerator, relativeRotation_info))
                {
                    ALICEVISION_LOG_INFO("Relative pose computation: (" << I << ", " << J << ") => FAILED");
                    continue;
                }

                relativePose_info.relativePose = geometry::Pose3(relativeRotation_info._relativeRotation, Vec3::Zero());
                relativePose_info.initial_residual_tolerance = relativeRotation_info._initialResidualTolerance;
                relativePose_info.found_residual_precision = relativeRotation_info._foundResidualPrecision;
                relativePose_info.vec_inliers = relativeRotation_info._inliers;
            }
            break;

            default:
            {
                ALICEVISION_LOG_DEBUG("Unknown relative rotation method: " << ERelativeRotationMethod_enumToString(_params.eRelativeRotationMethod));
            }
        }

        // If an existing prior on rotation exists, then make sure the found detected rotation is not stupid
        double weight = _params.rotationAveragingWeighting ? relativePose_info.vec_inliers.size() : 1.0;
        if (_sfmData.isPoseAndIntrinsicDefined(view_I) && _sfmData.isPoseAndIntrinsicDefined(view_J))
        {
            sfmData::CameraPose iTo = _sfmData.getAbsolutePose(view_I->getPoseId());
            sfmData::CameraPose jTo = _sfmData.getAbsolutePose(view_J->getPoseId());

            Eigen::Matrix3d iRo = iTo.getTransform().rotation();
            Eigen::Matrix3d jRo = jTo.getTransform().rotation();
            Eigen::Matrix3d jRi = jRo * iRo.transpose();

            Eigen::Matrix3d jRi_est = relativePose_info.relativePose.rotation();

            Eigen::AngleAxisd checker;
            checker.fromRotationMatrix(jRi_est * jRi.transpose());
            if (std::abs(radianToDegree(checker.angle())) > _params.maxAngleToPrior)
            {
                relativePose_info.relativePose = geometry::Pose3(jRi, Vec3::Zero());
                relativePose_info.vec_inliers.clear();
                weight = 1.0;
            }
        }

        // Sort all inliers by increasing ids
        if (!relativePose_info.vec_inliers.empty())
        {
            std::sort(relativePose_info.vec_inliers.begin(), relativePose_info.vec_inliers.end());

            size_t index = 0;
            size_t index_inlier = 0;

            const matching::MatchesPerDescType& matchesPerDesc = pair.second;
            for (const auto& matchesPerDescIt : matchesPerDesc)
            {
                const feature::EImageDescriberType descType = matchesPerDescIt.first;
                const matching::IndMatches& matches = matchesPerDescIt.second;

                for (const auto& match : matches)
                {
                    if (index_inlier >= relativePose_info.vec_inliers.size())
                    {
                        break;
                    }

                    size_t next_inlier = relativePose_info.vec_inliers[index_inlier];
                    if (index == next_inlier)
                    {
                        Vec2 pt1 = _featuresPerView->getFeatures(I, descType)[match._i].coords().cast<double>();
                        Vec2 pt2 = _featuresPerView->getFeatures(J, descType)[match._j].coords().cast<double>();

                        const feature::PointFeature& pI = _featuresPerView->getFeatures(I, descType)[match._i];
                        const feature::PointFeature& pJ = _featuresPerView->getFeatures(J, descType)[match._j];

                        const sfmData::Constraint2D constraint(I, sfmData::Observation(pt1, match._i, pI.scale()), J, sfmData::Observation(pt2, match._j, pJ.scale()), descType);

                        constraints2d.push_back(constraint);

                        ++index_inlier;
                    }

                    ++index;
                }
            }
        }

        // Add the relative rotation to the relative 'rotation' pose graph
        using namespace aliceVision::rotationAveraging;
        vec_relatives_R.emplace_back(view_I->getPoseId(), view_J->getPoseId(), relativePose_info.relativePose.rotation(), weight);
    } // for all relative pose
}

void ReconstructionEngine_rotation::filterMatches()
{
    // keep only the largest biedge connected subgraph
    const PairSet pairs = matching::getImagePairs(*_pairwiseMatches);
    const std::set<IndexT> set_remainingIds = graph::CleanGraph_KeepLargestBiEdge_Nodes<PairSet, IndexT>(pairs, _outputFolder);

    if (set_remainingIds.empty())
    {
        ALICEVISION_LOG_DEBUG("Invalid input image graph for panorama, no remaining match after filtering.");
    }

    KeepOnlyReferencedElement(set_remainingIds, *_pairwiseMatches);
}

} // namespace sfm
} // namespace aliceVision
