/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_LOGIT_LOSS_H_
#define DIFACTO_LOSS_LOGIT_LOSS_H_
#include <vector>
#include <cmath>
#include "difacto/base.h"
#include "difacto/loss.h"
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "common/spmv.h"
namespace difacto {

/**
 * \brief the logistic loss
 *
 * :math:`\ell(x,y,w) =  log(1 + exp(- y <w, x>))`
 *
 */
class LogitLoss : public Loss {
 public:
  LogitLoss() {}
  virtual ~LogitLoss() {}

  KWArgs Init(const KWArgs& kwargs) override {
    return kwargs;
  }

  /**
   * \brief perform prediction
   *
   *  pred += X * w
   *
   * @param data the data X
   * @param param input parameters
   * - param[0], real_t vector, the weights
   * - param[1], optional int vector, the weight positions
   * @param pred predict output, should be pre-allocated
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    CHECK_EQ(param.size(), 3);
    Predict(data,
            SArray<real_t>(param[0]),
            SArray<int>(param[1]),
            pred);
  }

  void Predict(const dmlc::RowBlock<unsigned>& data,
               const SArray<real_t>& weights,
               const SArray<int>& w_pos,
               SArray<real_t>* pred) {
    SArray<real_t> w = weights;
    SpMV::Times(data, w, pred, nthreads_, w_pos, {});
  }

  /*!
   * \brief compute the gradients
   *
   *   p = - y ./ (1 + exp (y .* pred));
   *   grad += X' * p;
   *
   * @param data the data X
   * @param param input parameters
   * - param[0], real_t vector, the predict output
   * - param[1], optional int vector, the gradient positions
   * @param grad the results, should be pre-allocated
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    CHECK_EQ(param.size(), 4);
    CalcGrad(data,
             SArray<real_t>(param[0]),
             SArray<int>(param[1]),
             SArray<real_t>(param[3]),
             grad);
  }

  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const SArray<real_t>& weights,
                const SArray<int>& w_pos,
                const SArray<real_t>& pred,
                SArray<real_t>* grad) {
    SArray<real_t> p; p.CopyFrom(pred);
    CHECK_EQ(p.size(), data.size);
    // p = ...
    CHECK_NOTNULL(data.label);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = data.label[i] > 0 ? 1 : -1;
      p[i] = - y / (1 + std::exp(y * p[i]));
    }

    // grad += ...
    SpMV::TransTimes(data, p, grad, nthreads_, {}, w_pos);
  }
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_LOGIT_LOSS_H_
