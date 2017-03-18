/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_LEARNER_H_
#define DIFACTO_SGD_SGD_LEARNER_H_
#include <string>
#include <vector>
#include "difacto/learner.h"
#include "difacto/loss.h"
#include "difacto/store.h"
#include "difacto/node_id.h"
#include "difacto/reporter.h"
#include "./sgd_utils.h"
#include "./sgd_updater.h"
#include "./sgd_param.h"


namespace difacto {

class SGDLearner : public Learner {
 public:
  SGDLearner() {
    store_ = nullptr;
    loss_ = nullptr;
    reporter_ = nullptr;
  }

  virtual ~SGDLearner() {
    delete loss_;
    delete store_;
    // reporter_ was deleted by shared_ptr
    // see Init()
    //delete reporter_;
  }
  KWArgs Init(const KWArgs& kwargs) override;

  void AddEpochEndCallback(const std::function<void(
      int epoch, const sgd::Progress& train, const sgd::Progress& val)>& callback) {
    epoch_end_callback_.push_back(callback);
  }

  SGDUpdater* GetUpdater() {
    return CHECK_NOTNULL(std::static_pointer_cast<SGDUpdater>(
        CHECK_NOTNULL(store_)->updater()).get());
  }

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets);

 private:
  void RunEpoch(int epoch, int job_type, sgd::Progress* prog);

  /** \brief save or load model */
  inline void SaveLoadModel(int type, int iter = -1) {
    sgd::Job job; std::string job_str;
    job.type = type; job.epoch = iter;
    job.SerializeToString(&job_str);
    tracker_->IssueAndWait(NodeID::kServerGroup, job_str);
  }

  /** \brief get the saved model name only for servers */
  inline std::string ModelName(const std::string& prefix, int iter) {
    std::string name = prefix;
    if (iter >= 0) name += "_iter-" + std::to_string(iter);
    return name + "_part-" + std::to_string(store_->Rank());
  }

  /** \brief save prediction to files only for workers */
  inline void SavePred(const SArray<real_t>& pred,
                       dmlc::real_t const* label = nullptr) const{
    std::string pred_name = param_.pred_out + "_part-" + std::to_string(store_->Rank());
    std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(pred_name.c_str(), "w"));
    dmlc::ostream os(fo.get());
    for (size_t i = 0; i < pred.size(); ++i) {
      if (label) os << label[i] << "\t";
      if(param_.pred_prob) {
        os << 1.0 / (1.0 + exp( - pred[i] )) << "\n";
      } else {
        os << pred[i] << "\n";
      }
    }
  }

  /**
   * \brief iterate on a part of a data
   *
   * it repeats the following steps
   *
   * 1. read batch_size examples
   * 2. preprogress data (map from uint64 feature index into continous ones)
   * 3. pull the newest model for this batch from the servers
   * 4. compute the gradients on this batch
   * 5. push the gradients to the servers to update the model
   *
   * to maximize the parallelization of i/o and computation, we uses three
   * threads here. they are asynchronized by callbacks
   *
   * a. main thread does 1 and 2
   * b. batch_tracker's thread does 3 once a batch is preprocessed
   * c. store_'s threads does 4 and 5 when the weight is pulled back
   */
  void IterateData(const sgd::Job& job, sgd::Progress* prog);

  real_t EvaluatePenalty(const SArray<real_t>& weight,
                         const SArray<int>& w_pos,
                         const SArray<int>& V_pos);
  void GetPos(const SArray<int>& len,
              SArray<int>* w_pos, SArray<int>* V_pos);

  /** \brief the model store*/
  Store* store_;
  /** \brief the loss*/
  Loss* loss_;
  /** \brief the reporter*/
  Reporter* reporter_;
  /** \brief parameters */
  SGDLearnerParam param_;
  // progress for reporter;
  sgd::Report_prog report_prog_;
  int blk_nthreads_ = DEFAULT_NTHREADS;
  double start_time_;
  bool do_embedding_ = false;

  std::vector<std::function<void(int epoch, const sgd::Progress& train,
                                 const sgd::Progress& val)>> epoch_end_callback_;
};

}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_LEARNER_H_
