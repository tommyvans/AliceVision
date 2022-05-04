// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/sfm/pipeline/ReconstructionEngine.hpp>
#include <aliceVision/sfm/pipeline/global/GlobalSfMRotationAveragingSolver.hpp>
#include <aliceVision/sfm/pipeline/global/GlobalSfMTranslationAveragingSolver.hpp>

#include <dependencies/htmlDoc/htmlDoc.hpp>

namespace aliceVision{
namespace sfm{


/**
 * Rotation Pipeline Reconstruction Engine.
 * Global approximations of rotations between cameras.
 */
class ReconstructionEngine_rotation : public ReconstructionEngine
{
public:
  struct Params
  {
      ERotationAveragingMethod eRotationAveragingMethod = ROTATION_AVERAGING_L2;
      ERelativeRotationMethod eRelativeRotationMethod = RELATIVE_ROTATION_FROM_E;
      bool rotationAveragingWeighting = true;
      double maxAngularError = 100.0;
      double maxAngleToPrior = 5.0;  //< max angle to input prior in degree
  };

  ReconstructionEngine_rotation(const sfmData::SfMData& sfmData,
                                const Params& params,
                                const std::string& outDirectory);

  ~ReconstructionEngine_rotation();

  void SetFeaturesProvider(feature::FeaturesPerView* featuresPerView);
  void SetMatchesProvider(matching::PairwiseMatches* provider);

  virtual bool process();

  /**
   * @brief Filter feature matches to keep only the largest biedge connected subgraph.
  */
  void filterMatches();

protected:
  /// Compute from relative rotations the global rotations of the camera poses
  bool Compute_Global_Rotations(const aliceVision::rotationAveraging::RelativeRotations& vec_relatives_R, HashMap<IndexT, Mat3>& map_globalR);

private:
  /// Compute relative rotations
  void Compute_Relative_Rotations(aliceVision::rotationAveraging::RelativeRotations& vec_relatives_R);

  // Parameter
  Params _params;

  // Data provider
  feature::FeaturesPerView* _featuresPerView;
  matching::PairwiseMatches* _pairwiseMatches;
};

} // namespace sfm
} // namespace aliceVision
