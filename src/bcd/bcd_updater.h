/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_BCD_BCD_UPDATER_H_
#define DIFACTO_BCD_BCD_UPDATER_H_
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include "difacto/updater.h"
#include "dmlc/parameter.h"
#include "dmlc/io.h"
#include "difacto/store.h"
#include "common/find_position.h"
#include "common/kv_match.h"
#include "common/kv_union.h"
#include "./bcd_utils.h"
namespace difacto {

struct BCDUpdaterParam : public dmlc::Parameter<BCDUpdaterParam> {
  int V_dim = 0;
  int tail_feature_filter;

  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;
  /** \brief the learning rate :math:`\eta` (or :math:`\alpha`) for :math:`w` */
  float lr;

  DMLC_DECLARE_PARAMETER(BCDUpdaterParam) {
    DMLC_DECLARE_FIELD(tail_feature_filter).set_default(4);
    DMLC_DECLARE_FIELD(l1).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_default(.01);
    DMLC_DECLARE_FIELD(lr).set_default(.9);
  }
};

class BCDUpdater : public Updater {
 public:
  BCDUpdater() { }
  virtual ~BCDUpdater() { }

  KWArgs Init(const KWArgs& kwargs) override {
    return param_.InitAllowUnknown(kwargs);
  }

  const BCDUpdaterParam& param() const { return param_; }

  void Load(dmlc::Stream* fi) override {}

  void Save(bool save_aux, dmlc::Stream *fo) const override {}

  void Report() override{};

  void Get(const SArray<feaid_t>& feaids,
           int value_type,
           SArray<real_t>* values,
           SArray<int>* offsets) override {
    if (value_type == Store::kFeaCount) {
      values->resize(feaids.size());
      KVMatch(feaids_, feacnt_, feaids, values);
    } else if (value_type == Store::kWeight) {
      if (weights_.empty()) InitWeights();
      values->resize(feaids.size() * (param_.V_dim+1));
      if (param_.V_dim == 0) {
        KVMatch(feaids_, w_delta_, feaids, values);
      } else {
        offsets->resize(feaids.size());
        SArray<int> pos; FindPosition(feaids_, feaids, &pos);
        int *os = offsets->data(); os[0] = 0;
        real_t* val = values->data();
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          int start = offsets_[pos[i]+1];
          int len = offsets_[pos[i]+1] - start;
          os[1] = os[0] + len;
          memcpy(val, w_delta_.data() + start, len * sizeof(real_t));
          val += len; ++os;
        }
        values->resize(os[1]);
      }
    } else {
      LOG(FATAL) << "...";
    }
  }


  void Update(const SArray<feaid_t>& feaids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& offsets) override {
    if (value_type == Store::kFeaCount) {
      feaids_ = feaids;
      feacnt_ = values;
    } else if (value_type == Store::kGradient) {
      if (weights_.empty()) InitWeights();
      SArray<int> pos; FindPosition(feaids_, feaids, &pos);
      if (offsets.empty()) {
        int k = 2;
        CHECK_EQ(values.size(), feaids.size()*k);
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          UpdateWeight(pos[i], values.data()+i*k, k);
        }
      } else {
        CHECK_EQ(offsets.size(), feaids.size());
        CHECK_EQ(offsets.back(), static_cast<int>(values.size()));
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          UpdateWeight(pos[i], values.data()+offsets[i], offsets[i+1]-offsets[i]);
        }
      }
    } else {
      LOG(FATAL) << "...";
    }
  }

 private:
  void InitWeights() {
    // remove tail features
    CHECK_EQ(feaids_.size(), feacnt_.size());
    SArray<feaid_t> filtered;
    for (size_t i = 0; i < feaids_.size(); ++i) {
      if (feacnt_[i] > param_.tail_feature_filter) {
        filtered.push_back(feaids_[i]);
      }
    }
    feaids_ = filtered;
    feacnt_.clear();

    // init weight
    CHECK_EQ(param_.V_dim, 0);
    weights_.resize(feaids_.size());
    w_delta_.resize(feaids_.size());
    bcd::Delta::Init(feaids_.size(), &delta_);
  }

  void UpdateWeight(int idx, real_t const* grad, int grad_len) {
    // update w
    CHECK_GE(grad_len, 2);

    real_t g = grad[0];
    real_t g_pos = g + param_.l1, g_neg = g - param_.l1;
    real_t u = grad[1] / param_.lr + 1e-10;
    int i = offsets_.size() ? offsets_[idx] : idx;
    real_t w = weights_[i];
    real_t d = - w;

    if (g_pos <= u * w) {
      d = - g_pos / u;
    } else if (g_neg >= u * w) {
      d = - g_neg / u;
    }
    d = std::min(delta_[idx], std::max(- delta_[idx], d));
    bcd::Delta::Update(d, &delta_[idx]);
    weights_[i] += d;
    w_delta_[i] = d;
  }

  BCDUpdaterParam param_;
  SArray<feaid_t> feaids_;
  SArray<real_t> feacnt_;
  SArray<real_t> weights_;
  SArray<real_t> w_delta_;
  SArray<int> offsets_;
  SArray<real_t> delta_;
};



}  // namespace difacto


#endif  // DIFACTO_BCD_BCD_UPDATER_H_
