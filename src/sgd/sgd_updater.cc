/**
 * Copyright (c) 2015 by Contributors
 */
#include <string.h>
#include "./sgd_updater.h"
#include "difacto/store.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDUpdaterParam);

KWArgs SGDUpdater::Init(const KWArgs& kwargs) {
  auto remain = param_.InitAllowUnknown(kwargs);
  CHECK_GT(param_.V_dim, 0);
  CHECK_GT(param_.field_num, 0);
  feat_dim = param_.V_dim * param_.field_num;
  return remain;
}

void SGDUpdater::Evaluate(sgd::Progress* prog) const {
  real_t objv = 0;
  size_t nnz = 0;
  mu_.lock();
  for (const auto& it : model_) {
    const auto& e = it.second;
    for (int i = 0; i < feat_dim; ++i) {
      if (e.V[i] != 0) {
        objv += .5 * param_.l2 * e.V[i] * e.V[i];
        nnz += 1;
      }
    }
  }
  mu_.unlock();
  prog->penalty = objv;
  prog->nnz_w = nnz;
}

void SGDUpdater::Get(const SArray<feaid_t>& fea_ids,
                     int val_type,
                     SArray<real_t>* weights,
                     SArray<int>* lens) {
  CHECK_EQ(val_type, Store::kWeight);
  size_t size = fea_ids.size();
  weights->resize(size * feat_dim);
  if (lens) lens->resize(size);
  int p = 0;
  for (size_t i = 0; i < size; ++i) {
    //mu_.lock();
    auto& e = model_[fea_ids[i]];
    //mu_.unlock();
    memcpy(weights->data()+p, e.V, feat_dim * sizeof(real_t));
    p += feat_dim;
    if (lens) (*lens)[i] = feat_dim;
  }
  weights->resize(p);
}

void SGDUpdater::Update(const SArray<feaid_t>& fea_ids,
                        int value_type,
                        const SArray<real_t>& values,
                        const SArray<int>& lens) {
  if (value_type == Store::kFeaCount) {
    CHECK_EQ(fea_ids.size(), values.size());
    for (size_t i = 0; i < fea_ids.size(); ++i) {
      //mu_.lock();
      auto& e = model_[fea_ids[i]];
      //mu_.unlock();
      e.fea_cnt += values[i];
      if (e.V == nullptr && e.fea_cnt > param_.V_threshold) {
        InitV(&e);
      }
    }
  } else if (value_type == Store::kGradient) {
    size_t size = fea_ids.size();
    CHECK_EQ(values.size(), size * feat_dim);
    if (!lens.empty()) CHECK_EQ(lens.size(), size);
    int p = 0;
    real_t* v = values.data();
    for (size_t i = 0; i < size; ++i) {
      //mu_.lock();
      auto& e = model_[fea_ids[i]];
      //mu_.unlock();
      if (!lens.empty()) CHECK_EQ(lens[i], feat_dim);
      CHECK(e.V != nullptr) << fea_ids[i];
      UpdateV(v+p, &e);
      p += feat_dim;
    }
    CHECK_EQ(static_cast<size_t>(p), values.size());
  } else {
    LOG(FATAL) << "UNKNOWN value_type.....";
  }
}

/*
void SGDUpdater::UpdateW(real_t gw, SGDEntry* e) {
  real_t sg = e->sqrt_g;
  real_t w = e->w;
  // update sqrt_g
  gw += w * param_.l2;
  e->sqrt_g = sqrt(sg * sg + gw * gw);
  // update z
  e->z -= gw - (e->sqrt_g - sg) / param_.lr * w;
  // update w by soft shrinkage
  real_t z = e->z;
  real_t l1 = param_.l1;
  if (z <= l1 && z >= - l1) {
    e->w = 0;
  } else {
    real_t eta = (param_.lr_beta + e->sqrt_g) / param_.lr;
    e->w = (z > 0 ? z - l1 : z + l1) / eta;
  }
  // update statistics
  if (w == 0 && e->w != 0) {
    ++ new_w;
    if (param_.V_dim > 0 && e->V == nullptr && e->fea_cnt > param_.V_threshold) {
      InitV(e);
    }
  } else if (w != 0 && e->w == 0) {
    -- new_w;
  }
}
*/

void SGDUpdater::UpdateV(real_t const* gV, SGDEntry* e) {
  for (int i = 0; i < feat_dim; ++i) {
    real_t g = gV[i] + param_.V_l2 * e->V[i];
    real_t cg = e->V[i+feat_dim];
    e->Z[i] = sqrt(cg * cg + g * g);
    float eta = param_.V_lr / (e->Z[i] + param_.V_lr_beta);
    e->V[i] -= eta * g;
  }
}

void SGDUpdater::InitV(SGDEntry* e) {
  e->V = new real_t[feat_dim];
  e->Z = new real_t[feat_dim];
  for (int i = 0; i < feat_dim; ++i) {
    e->V[i] = (rand_r(&param_.seed) / (real_t)RAND_MAX - 0.5) * param_.V_init_scale;
  }
  memset(e->Z, 0, feat_dim * sizeof(real_t));
  e->size = feat_dim;
  e->nnz = feat_dim;
}

}  // namespace difacto
