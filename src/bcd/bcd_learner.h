/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_BCD_BCD_LEARNER_H_
#define DIFACTO_BCD_BCD_LEARNER_H_
#include <vector>
#include <string>
#include "difacto/learner.h"
#include "difacto/store.h"
#include "data/data_store.h"
#include "data/tile_store.h"
#include "data/tile_builder.h"
#include "common/learner_utils.h"
#include "./bcd_param.h"
#include "./bcd_utils.h"
#include "loss/logit_loss_delta.h"
namespace difacto {

class BCDLearner : public Learner {
 public:
  BCDLearner() {}
  virtual ~BCDLearner() {
    delete model_store_;
    delete tile_store_;
    delete loss_;
  }

  KWArgs Init(const KWArgs& kwargs) override;

  void AddEpochEndCallback(
      const std::function<void(int epoch, const std::vector<real_t>& prog)>& callback) {
    epoch_end_callback_.push_back(callback);
  }

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets) override;

 private:
  void IssueJobAndWait(int node_group, const bcd::Job& job, std::vector<real_t>*
                       rets = nullptr) {
    std::string args; job.SerializeToString(&args);
    SendJobAndWait(node_group, args, tracker_, rets);
  }
  void PrepareData(std::vector<real_t>* fea_stats);

  void BuildFeatureMap(const std::vector<Range>& feablk_ranges);

  void IterateData(const std::vector<int>& feablks, std::vector<real_t>* progress);

  /**
   * \brief iterate a feature block
   *
   * the logic is as following
   *
   * 1. calculate gradident
   * 2. push gradients to servers, so servers will update the weight
   * 3. once the push is done, pull the changes for the weights back from
   *    the servers
   * 4. once the pull is done update the prediction
   *
   * however, two things make the implementation is not so intuitive.
   *
   * 1. we need to iterate the data block one by one for both calcluating
   * gradient and update prediction
   * 2. we used callbacks to avoid to be blocked by the push and pull.
   *
   * NOTE: once cannot iterate on the same block before it is actually finished.
   *
   * @param blk_id
   * @param on_complete will be called when actually finished
   */
  void IterateFeablk(int blk_id,
                     const std::function<void()>& on_complete,
                     std::vector<real_t>* progress);

  void CalcGrad(int rowblk_id, int colblk_id,
                const SArray<int>& grad_offset,
                SArray<real_t>* grad);

  void UpdtPred(int rowblk_id, int colblk_id,
                const SArray<int> delta_w_offset,
                const SArray<real_t> delta_w,
                std::vector<real_t>* progress);

  /** \brief the current epoch */
  int epoch_ = 0;
  int ntrain_blks_ = 0;
  int nval_blks_ = 0;

  /** \brief the model store*/
  Store* model_store_ = nullptr;
  /** \brief the loss function */
  Loss* loss_ = nullptr;
  /** \brief data store */
  TileStore* tile_store_ = nullptr;
  TileBuilder* tile_builder_ = nullptr;

  /** \brief parameters */
  BCDLearnerParam param_;

  /** \brief data associated with a feature block */
  struct FeaBlk {
    SArray<feaid_t> feaids;
    Range pos;
    SArray<real_t> delta;
    SArray<int> model_offset;
  };
  std::vector<FeaBlk> feablks_;

  SArray<feaid_t> feaids_;

  std::vector<SArray<real_t>> pred_;

  std::vector<std::function<void(
      int epoch, const std::vector<real_t> & prog)>> epoch_end_callback_;
};

}  // namespace difacto
#endif  // DIFACTO_BCD_BCD_LEARNER_H_
