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

#include <aliceVision/sfm/pipeline/rotation/relativeRotations.hpp>

namespace aliceVision{
namespace sfm{

/**
 * Panorama Pipeline Reconstruction Engine.
 * The method is based on the Global SfM but with no translations between cameras.
 */
class ReconstructionEngine_panorama : public ReconstructionEngine
{
public:
  struct Params
  {
      bool lockAllIntrinsics = false;
      bool intermediateRefineWithFocal = false; //< intermediate refine with rotation+focal
      bool intermediateRefineWithFocalDist = false; //< intermediate refine with rotation+focal+distortion
  };
  ReconstructionEngine_panorama(const sfmData::SfMData& sfmData,
                                const Params& params,
                                const std::string& outDirectory,
                                const std::string& loggingFile = "");

  ~ReconstructionEngine_panorama();

  void SetFeaturesProvider(feature::FeaturesPerView* featuresPerView);
  void SetMatchesProvider(matching::PairwiseMatches* provider);
  
  virtual bool process();
  bool buildLandmarks();

public:
  /// Adjust the scene (& remove outliers)
  bool Adjust();

private:
  // Logger
  std::shared_ptr<htmlDocument::htmlDocumentStream> _htmlDocStream;
  std::string _loggingFile;

  // Parameter
  Params _params;

  // Data provider
  feature::FeaturesPerView* _featuresPerView;
  matching::PairwiseMatches* _pairwiseMatches;

};

} // namespace sfm
} // namespace aliceVision
