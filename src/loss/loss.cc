/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/loss.h"
#include "./ffm_loss.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(FFMLossParam);

Loss* Loss::Create(const std::string& type, int nthreads) {
  Loss* loss = nullptr;
  if (type == "ffm") {
    loss = new FFMLoss();
  } else {
    LOG(FATAL) << "unknown loss type";
  }
  loss->set_nthreads(nthreads);
  return loss;
}

}  // namespace difacto
