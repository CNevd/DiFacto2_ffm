/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "./sgd/sgd_param.h"
#include "./sgd/sgd_learner.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDLearnerParam);

Learner* Learner::Create(const std::string& type) {
  if (type == "sgd") {
    return new SGDLearner();
  } else {
    LOG(FATAL) << "unknown learner type: " << type;
  }
  return nullptr;
}

KWArgs Learner::Init(const KWArgs& kwargs) {
  // init job tracker
  tracker_ = Tracker::Create();
  auto remain = tracker_->Init(kwargs);
  using namespace std::placeholders;
  tracker_->SetExecutor(std::bind(&Learner::Process, this, _1, _2));
  return remain;
}

}  // namespace difacto
