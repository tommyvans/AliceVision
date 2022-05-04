// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ReconstructionEngine_panorama.hpp"
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/graph/connectedComponent.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/stl/stl.hpp>
#include <aliceVision/multiview/essential.hpp>
#include <aliceVision/track/TracksBuilder.hpp>
#include <aliceVision/config.hpp>

#include <aliceVision/multiview/triangulation/triangulationDLT.hpp>
#include <aliceVision/multiview/triangulation/Triangulation.hpp>
#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/relativePose/Homography4PSolver.hpp>
#include <aliceVision/multiview/relativePose/HomographyError.hpp>
#include <aliceVision/multiview/relativePose/EssentialKernel.hpp>
#include <aliceVision/multiview/relativePose/Rotation3PSolver.hpp>

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/numeric/projection.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/robustEstimation/IRansacKernel.hpp>
#include <aliceVision/robustEstimation/conditioning.hpp>

#include <aliceVision/matching/supportEstimation.hpp>

#include <aliceVision/sfm/BundleAdjustmentPanoramaCeres.hpp>


#include <dependencies/htmlDoc/htmlDoc.hpp>

#include <boost/progress.hpp>

#ifdef _MSC_VER
#pragma warning( once : 4267 ) //warning C4267: 'argument' : conversion from 'size_t' to 'const int', possible loss of data
#endif

namespace aliceVision {
namespace sfm {

using namespace aliceVision::camera;
using namespace aliceVision::geometry;
using namespace aliceVision::feature;
using namespace aliceVision::sfmData;


ReconstructionEngine_panorama::ReconstructionEngine_panorama(const SfMData& sfmData,
                                                             const ReconstructionEngine_panorama::Params& params,
                                                             const std::string& outDirectory,
                                                             const std::string& loggingFile)
  : ReconstructionEngine(sfmData, outDirectory)
  , _params(params)
  , _loggingFile(loggingFile)
{
  if(!_loggingFile.empty())
  {
    // setup HTML logger
    _htmlDocStream = std::make_shared<htmlDocument::htmlDocumentStream>("PanoramaReconstructionEngine SFM report.");
    _htmlDocStream->pushInfo(htmlDocument::htmlMarkup("h1", std::string("ReconstructionEngine_panorama")));
    _htmlDocStream->pushInfo("<hr>");
    _htmlDocStream->pushInfo( "Dataset info:");
    _htmlDocStream->pushInfo( "Views count: " + htmlDocument::toString( sfmData.getViews().size()) + "<br>");
  }
}

ReconstructionEngine_panorama::~ReconstructionEngine_panorama()
{
  if(!_loggingFile.empty())
  {
    // Save the reconstruction Log
    std::ofstream htmlFileStream(_loggingFile.c_str());
    htmlFileStream << _htmlDocStream->getDoc();
  }
}

void ReconstructionEngine_panorama::SetFeaturesProvider(feature::FeaturesPerView* featuresPerView)
{
  _featuresPerView = featuresPerView;
}

void ReconstructionEngine_panorama::SetMatchesProvider(matching::PairwiseMatches* provider)
{
  _pairwiseMatches = provider;
}


bool ReconstructionEngine_panorama::process()
{
  return true;
}

// Adjust the scene (& remove outliers)
bool ReconstructionEngine_panorama::Adjust()
{
  BundleAdjustmentPanoramaCeres::CeresOptions options;
  options.summary = true;
  
  // Start bundle with rotation only
  BundleAdjustmentPanoramaCeres BA(options);
  bool success = BA.adjust(_sfmData, BundleAdjustmentPanoramaCeres::REFINE_ROTATION);
  if(success)
  {
    ALICEVISION_LOG_INFO("Rotations successfully refined.");
  }
  else
  {
    ALICEVISION_LOG_INFO("Failed to refine the rotations only.");
    return false;
  }

  if(_params.lockAllIntrinsics)
  {
      // no not modify intrinsic camera parameters
      return true;
  }

  if(_params.intermediateRefineWithFocal)
  {
      success = BA.adjust(_sfmData, BundleAdjustmentPanoramaCeres::REFINE_ROTATION |
                                    BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_FOCAL);
      if(success)
      {
        ALICEVISION_LOG_INFO("Bundle successfully refined: Rotation + Focal");
      }
      else
      {
        ALICEVISION_LOG_INFO("Failed to refine: Rotation + Focal");
          return false;
      }
  }
  if(_params.intermediateRefineWithFocalDist)
  {
      success = BA.adjust(_sfmData, BundleAdjustmentPanoramaCeres::REFINE_ROTATION |
                                    BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_FOCAL |
                                    BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_DISTORTION);
      if(success)
      {
        ALICEVISION_LOG_INFO("Bundle successfully refined: Rotation + Focal + Distortion");
      }
      else
      {
        ALICEVISION_LOG_INFO("Failed to refine: Rotation + Focal + Distortion");
          return false;
      }
  }

  // Minimize All
  success = BA.adjust(_sfmData, BundleAdjustmentPanoramaCeres::REFINE_ROTATION |
                                BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_FOCAL |
                                BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_DISTORTION |
                                BundleAdjustmentPanoramaCeres::REFINE_INTRINSICS_OPTICALOFFSET_ALWAYS);
  if(success)
  {
      ALICEVISION_LOG_INFO("Bundle successfully refined: Rotation + Focal + Optical Center + Distortion");
  }
  else
  {
      ALICEVISION_LOG_INFO("Failed to refine: Rotation + Focal + Distortion + Optical Center");
      return false;
  }

  return true;
}

bool ReconstructionEngine_panorama::buildLandmarks()
{
  // Remove all landmarks
  _sfmData.getLandmarks().clear();

  size_t count = 0;
  for (const sfmData::Constraint2D & c : _sfmData.getConstraints2D())
  {
    // Retrieve camera parameters
    const sfmData::View & v1 = _sfmData.getView(c.ViewFirst);
    const std::shared_ptr<camera::IntrinsicBase> cam1 = _sfmData.getIntrinsicsharedPtr(v1.getIntrinsicId());
    const sfmData::CameraPose pose1 = _sfmData.getPose(v1);
    const Vec3 wpt1 = cam1->backproject(c.ObservationFirst.x, true, pose1.getTransform(), 1.0);

    const sfmData::View & v2 = _sfmData.getView(c.ViewSecond);
    const std::shared_ptr<camera::IntrinsicBase> cam2 = _sfmData.getIntrinsicsharedPtr(v2.getIntrinsicId());
    const sfmData::CameraPose pose2 = _sfmData.getPose(v2);
    const Vec3 wpt2 = cam2->backproject(c.ObservationSecond.x, true, pose2.getTransform(), 1.0);

    // Store landmark
    Landmark l;
    l.descType = c.descType;
    l.observations[c.ViewFirst] = c.ObservationFirst;
    l.observations[c.ViewSecond] = c.ObservationSecond;
    l.X = (wpt1 + wpt2) * 0.5;

    _sfmData.getLandmarks()[count++] = l;
  }

  return true;
}

} // namespace sfm
} // namespace aliceVision

