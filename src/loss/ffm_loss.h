/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_FM_LOSS_H_
#define DIFACTO_LOSS_FM_LOSS_H_
#include <vector>
#include <cmath>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "difacto/loss.h"
#include "common/spmv.h"
#include "common/spmm.h"
namespace difacto {
/**
 * \brief parameters for FM loss
 */
struct FFMLossParam : public dmlc::Parameter<FFMLossParam> {
  /**
   * \brief the embedding dimension
   */
  int V_dim;
  int field_num;
  DMLC_DECLARE_PARAMETER(FFMLossParam) {
    DMLC_DECLARE_FIELD(V_dim).set_range(0, 10000);
    DMLC_DECLARE_FIELD(field_num).set_range(0, 10000);
  }
};
/**
 * \brief the factonization machine loss
 * :math:`f(x) = \langle w, x \rangle + \frac{1}{2} \|V x\|_2^2 - \sum_{i=1}^d x_i^2 \|V_i\|^2_2`
 */
class FFMLoss : public Loss {
 public:
  FFMLoss() {}
  virtual ~FFMLoss() {}

  KWArgs Init(const KWArgs& kwargs) override {
    auto remain = param_.InitAllowUnknown(kwargs);
    feat_num = param_.V_dim * param_.field_num;
    return remain;
  }
  /**
   * \brief perform prediction
   *
   * @param data the data
   * @param param input parameters
   * - param[0], real_t vector, the weights
   * - param[1], int vector, the V positions
   * @param pred predict output, should be pre-allocated
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    CHECK_EQ(param.size(), 2);
    Predict(data,
            SArray<real_t>(param[0]),
            SArray<int>(param[1]),
            pred);
  }

  void Predict(const dmlc::RowBlock<unsigned>& data,
               const SArray<real_t>& weights,
               const SArray<int>& V_pos,
               SArray<real_t>* pred) {
    SArray<real_t> w = weights;
    int V_dim = param_.V_dim;
#pragma omp parallel num_threads(nthreads_)
    {
      Range rg = Range(0, data.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (data.offset[i] == data.offset[i+1]) continue;
        real_t p = 0.;
        for (size_t j1 = data.offset[i]; j1 < data.offset[i+1]; ++j1) {
          int ind1 = data.index[j1];
          if (V_pos[ind1] < 0) continue;
          for (size_t j2 = data.offset[i]+1; j2 < data.offset[i+1]; ++j2) {
            int ind2 = data.index[j2];
            if (V_pos[ind2] < 0) continue;
            int f1 = data.field[j1], f2 = data.field[j2];
            real_t ww = 0.;
            for (int k = 0; k < V_dim; ++k) {
              ww += weights[ind1 * feat_num + f1 * V_dim + k] * \
                    weights[ind2 * feat_num + f2 * V_dim + k];
            }
            if (data.value) {
              real_t vv = data.value[j1] * data.value[j2];
              p += ww * vv;
            } else {
              p += ww;
            }
          }
        }
        (*pred)[i] = p > 20 ? 20 : (p < -20 ? -20 : p);
      }
    }
  }

  /*!
   * \brief compute the gradients
   *
   *   p = - y ./ (1 + exp (y .* pred));
   *
   * @param data the data
   * @param param input parameters
   * - param[0], real_t vector, the weights
   * - param[1], int vector, the V positions
   * - param[2], real_t vector, the predict output
   * @param grad the results
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    CHECK_EQ(param.size(), 3);
    CalcGrad(data,
             SArray<real_t>(param[0]),
             SArray<int>(param[1]),
             SArray<real_t>(param[2]),
             grad);
  }

  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const SArray<real_t>& weights,
                const SArray<int>& V_pos,
                const SArray<real_t>& pred,
                SArray<real_t>* grad) {
    // p = ...
    SArray<real_t> p; p.CopyFrom(pred);
    CHECK_EQ(p.size(), data.size);
    int V_dim = param_.V_dim;
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = data.label[i] > 0 ? 1 : -1;
      p[i] = - y / (1 + std::exp(y * p[i]));
      if (data.weight) p[i] *= data.weight[i];
    }

#pragma omp parallel num_threads(nthreads_)
    {
      Range rg = Range(0, data.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (data.offset[i] == data.offset[i+1]) continue;
        for (size_t j1 = data.offset[i]; j1 < data.offset[i+1]; ++j1) {
          int ind1 = data.index[j1];
          if (V_pos[ind1] < 0) continue;
          for (size_t j2 = data.offset[i]+1; j2 < data.offset[i+1]; ++j2) {
            int ind2 = data.index[j2];
            if (V_pos[ind2] < 0) continue;
            int idx1 = ind1 * feat_num + data.field[j1] * V_dim;
            int idx2 = ind2 * feat_num + data.field[j2] * V_dim;
            for (int k = 0; k < V_dim; ++k) {
              if (data.value) {
                real_t vv = data.value[j1] * data.value[j2];
                (*grad)[idx1 + k] += weights[idx2 + k] * p[i] * vv;
                (*grad)[idx2 + k] += weights[idx1 + k] * p[i] * vv;
              } else {
                (*grad)[idx1 + k] += weights[idx2 + k] * p[i];
                (*grad)[idx2 + k] += weights[idx1 + k] * p[i];
              }
            }
          }
        }
      }
    }
  }

 private:
  FFMLossParam param_;
  int feat_num = 0;
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_FM_LOSS_H_
