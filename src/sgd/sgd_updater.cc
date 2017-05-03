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
  coef = 1.0f / sqrt(param_.V_dim);
  distribution = std::uniform_real_distribution<float>(-param_.V_init_scale, param_.V_init_scale);
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
  lens->resize(size);
  int p = 0;
  for (size_t i = 0; i < size; ++i) {
    mu_.lock();
    auto& e = model_[fea_ids[i]];
    mu_.unlock();
    if (e.empty()) {
      (*lens)[i] = 0;
    } else {
      memcpy(weights->data()+p, e.V, feat_dim * sizeof(real_t));
      p += feat_dim;
      (*lens)[i] = feat_dim;
    }
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
      mu_.lock();
      auto& e = model_[fea_ids[i]];
      mu_.unlock();
      e.fea_cnt += values[i];
      if (e.V == nullptr && e.fea_cnt > param_.V_threshold) {
        InitV(&e);
      }
    }
  } else if (value_type == Store::kGradient) {
    size_t size = fea_ids.size();
    CHECK_EQ(lens.size(), size);
    int p = 0;
    real_t* v = values.data();
    for (size_t i = 0; i < size; ++i) {
      if (lens[i] == 0) continue;
      mu_.lock();
      auto& e = model_[fea_ids[i]];
      mu_.unlock();
      CHECK_EQ(lens[i], feat_dim);
      CHECK(e.V != nullptr) << fea_ids[i];
      UpdateV(v+p, &e);
      p += feat_dim;
    }
    CHECK_EQ(static_cast<size_t>(p), values.size());
  } else {
    LOG(FATAL) << "UNKNOWN value_type.....";
  }
}

void SGDUpdater::UpdateV(real_t const* gV, SGDEntry* e) {
  int nnz = e->nnz;
  for (int i = 0; i < feat_dim; ++i) {
    real_t sg = e->Z[i];
    real_t vi = e->V[i];

    // update sqrt_g
    real_t gv = gV[i] + vi * param_.l2;
    e->Z[i] = sqrt(sg * sg + gv * gv);
    e->V[i] -= param_.lr * e->Z[i] * gv;
   
    // FTRL 
    /*
    e->Z[i + feat_dim] -= gv - (e->Z[i] - sg) / param_.lr * vi;

    real_t z = e->Z[i + feat_dim];
    real_t l1 = param_.l1;
    if (z <= l1 && z >= - l1) {
      e->V[i] = 0;
    } else {
      real_t eta = (param_.lr_beta + e->Z[i]) / param_.lr + param_.l2;
      e->V[i] = (z > 0 ? z - l1 : z + l1) / eta;
    }
    */

    // update statistics
    if (vi == 0 && e->V[i] != 0) {
      ++ e->nnz;
    } else if (vi != 0 && e->V[i] == 0) {
      -- e->nnz;
    }
  }
  new_w += (e->nnz - nnz);
}

void SGDUpdater::InitV(SGDEntry* e) {
  e->V = new real_t[feat_dim];
  e->Z = new real_t[feat_dim*2];
  for (int i = 0; i < feat_dim; ++i) {
    e->V[i] = coef * distribution(generator);
    if (e->V[i] != 0) e->nnz += 1;
  }
  memset(e->Z, 1.0, feat_dim * sizeof(real_t));
  memset(e->Z + feat_dim, 0, feat_dim * sizeof(real_t));
  e->size = feat_dim;
  new_w += e->nnz;
}

}  // namespace difacto
