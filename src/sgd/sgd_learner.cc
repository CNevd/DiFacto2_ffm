/**
 *  Copyright (c) 2015 by Contributors
 */
#include "./sgd_learner.h"
#include <stdlib.h>
#include <memory>
#include <thread>
#include <vector>
#include <utility>
#include "dmlc/timer.h"
#include "reader/batch_reader.h"
#include "tracker/async_local_tracker.h"
#include "data/shared_row_block_container.h"
#include "data/row_block.h"
#include "data/localizer.h"
#include "loss/bin_class_metric.h"
namespace difacto {

/** \brief struct to hold info for a batch job */
struct BatchJob {
  int type;
  SArray<feaid_t> feaids;
  SharedRowBlockContainer<unsigned> data;
};

KWArgs SGDLearner::Init(const KWArgs& kwargs) {
  // init tracker
  auto remain = Learner::Init(kwargs);
  // init param
  remain = param_.InitAllowUnknown(remain);
  // init reporter
  reporter_ = Reporter::Create();
  remain = reporter_->Init(remain);
  // init updater
  auto updater = new SGDUpdater();
  remain = updater->Init(remain);
  remain.push_back(std::make_pair("V_dim", std::to_string(updater->param().V_dim)));
  // init do embedding
  if (updater->param().V_dim > 0) do_embedding_ = true;
  // init store
  store_ = Store::Create();
  store_->SetUpdater(std::shared_ptr<Updater>(updater));
  store_->SetReporter(std::shared_ptr<Reporter>(reporter_));
  remain = store_->Init(remain);
  // init loss
  loss_ = Loss::Create(param_.loss, blk_nthreads_);
  remain = loss_->Init(remain);
  
  return remain;
}

void SGDLearner::RunScheduler() { 
  real_t pre_loss = 0, pre_val_auc = 0;
  int k = 0;
  start_time_ = dmlc::GetTime();

  // load model
  if (param_.model_in.size()) {
    if (param_.load_epoch > 0) {
      LOG(INFO) << "Loading model from epoch " << param_.load_epoch;
      SaveLoadModel(sgd::Job::kLoadModel, param_.load_epoch);
      k = param_.load_epoch + 1;
    } else {
      LOG(INFO) << "loading lastest model...";
      SaveLoadModel(sgd::Job::kLoadModel);
    }
  }

  if (0) {
    CHECK(param_.model_in.size()) << "Prediction needs model_in";
    sgd::Progress pred_prog;
    LOG(INFO) << "Start predicting...";
    RunEpoch(k, sgd::Job::kPrediction, &pred_prog);
    LOG(INFO) << "Prediction: " << pred_prog.TextString();
    Stop();
    return;
  }

  for (; k < param_.max_num_epochs; ++k) {
    sgd::Progress train_prog;
    LOG(INFO) << "Start epoch " << k;
    RunEpoch(k, sgd::Job::kTraining, &train_prog);
    LOG(INFO) << "Epoch[" << k << "] Training: " << train_prog.TextString();

    sgd::Progress val_prog;
    if (param_.data_val.size()) {
      RunEpoch(k, sgd::Job::kValidation, &val_prog);
      LOG(INFO) << "Epoch[" << k << "] Validation: " << val_prog.TextString();
    }
    for (const auto& cb : epoch_end_callback_) cb(k, train_prog, val_prog);

    // stop criteria
    real_t eps = fabs(train_prog.loss - pre_loss) / pre_loss;
    if (eps < param_.stop_rel_objv) {
      LOG(INFO) << "Change of loss [" << eps << "] < stop_rel_objv ["
                << param_.stop_rel_objv << "]";
      break;
    }
    if (val_prog.auc > 0) {
      eps = (val_prog.auc - pre_val_auc) / val_prog.nrows;
      if (eps < param_.stop_val_auc) {
        LOG(INFO) << "Change of validation AUC [" << eps << "] < stop_val_auc ["
                  << param_.stop_val_auc << "]";
        break;
      }
    }
    if (k+1 >= param_.max_num_epochs) {
      LOG(INFO) << "Reach maximal number of epochs " << param_.max_num_epochs;
      break;
    }
    pre_loss = train_prog.loss;
    pre_val_auc = val_prog.auc;
  }

  // Save last model
  if (param_.model_out.size()) {
    LOG(INFO) << "Saving the final model...";
    SaveLoadModel(sgd::Job::kSaveModel);
    LOG(INFO) << "Save model finished";
  }
  Stop();
}

void SGDLearner::RunEpoch(int epoch, int job_type, sgd::Progress* prog) {
  // progress merger
  tracker_->SetMonitor(
      [this, prog](int node_id, const std::string& rets) {
        prog->Merge(rets);
      });

  // progress reporter
  reporter_->SetMonitor(
      [this](int node_id, const std::string& rets) {
        report_prog_.prog.Merge(rets);
      });

  // Start Dispatch
  int n = store_->NumWorkers() * param_.num_jobs_per_epoch;
  tracker_->StartDispatch(n, job_type, epoch);

  // wait and report
  while (tracker_->NumRemains()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(param_.report_interval * 1000));
    if (job_type == sgd::Job::kTraining) {
      printf("%5.0lf  %s\n", dmlc::GetTime() - start_time_, report_prog_.PrintStr().c_str());
      fflush(stdout);
    }
  }
}

void SGDLearner::GetPos(const SArray<int>& len,
                        SArray<int>* w_pos, SArray<int>* V_pos) {
  size_t n = len.size();
  w_pos->resize(n);
  V_pos->resize(n);
  int* w = w_pos->data();
  int* V = V_pos->data();
  int p = 0;
  for (size_t i = 0; i < n; ++i) {
    int l = len[i];
    w[i] = l == 0 ? -1 : p;
    V[i] = l > 1 ? p+1 : -1;
    p += l;
  }
}

void SGDLearner::Process(const std::string& args, std::string* rets) {
  if (args.empty()) return;
  using sgd::Job;
  sgd::Progress prog;
  Job job; job.ParseFromString(args);
  switch(job.type) {
    case Job::kTraining:
    case Job::kValidation:
    case Job::kPrediction: {
      IterateData(job, &prog);
      break;
    }
    case Job::kEvaluation: {
      GetUpdater()->Evaluate(&prog);
      break;
    }
    case Job::kLoadModel: {
      std::string filename = ModelName(param_.model_in, job.epoch);
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(filename.c_str(), "r"));
      GetUpdater()->Load(fi.get());
      break;
    }
    case Job::kSaveModel: {
      std::string filename = ModelName(param_.model_out, job.epoch);
      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(filename.c_str(), "w"));
      GetUpdater()->Save(param_.has_aux, fo.get());
      break;
    }
  }
  prog.SerializeToString(rets);
}

void SGDLearner::IterateData(const sgd::Job& job, sgd::Progress* progress) {
  AsyncLocalTracker<BatchJob> batch_tracker;
  batch_tracker.SetExecutor(
      [this, progress](const BatchJob& batch,
                       const std::function<void()>& on_complete,
                       std::string* rets) {
        // use potiners here in order to copy into the callback
        SArray<real_t>* values = new SArray<real_t>();
        SArray<int>* lengths = do_embedding_ ? new SArray<int>() : nullptr;
        auto pull_callback = [this, batch, values, lengths, progress, on_complete]() {
          // eval loss
          auto data = batch.data.GetBlock();
          progress->nrows += data.size;
          SArray<real_t> pred(data.size);
          SArray<int> w_pos, V_pos;
          if (lengths) GetPos(*lengths, &w_pos, &V_pos);
          std::vector<SArray<char>> inputs = {
            SArray<char>(*values), SArray<char>(w_pos), SArray<char>(V_pos)};
          CHECK_NOTNULL(loss_)->Predict(data, inputs, &pred);
          auto loss = loss_->Evaluate(batch.data.label.data(), pred);
          progress->loss += loss;
          // eval penalty
          //progress->penalty += EvaluatePenalty(*values, w_pos, V_pos);

          // auc, ...
          BinClassMetric metric(batch.data.label.data(), pred.data(),
                                pred.size(), blk_nthreads_);
          auto auc = metric.AUC();
          progress->auc += auc;

          if (batch.type == sgd::Job::kPrediction && param_.pred_out.size()) {
            SavePred(pred, batch.data.label.data());
          }

          // calculate the gradients
          if (batch.type == sgd::Job::kTraining) {
            // report progress to SCH first
            sgd::Progress report_prog; std::string rets;
            report_prog.nrows = data.size;
            report_prog.loss = loss; report_prog.auc = auc;
            report_prog.SerializeToString(&rets);
            reporter_->Report(rets);

            SArray<real_t> grads(values->size());
            inputs.push_back(SArray<char>(pred));
            loss_->CalcGrad(data, inputs, &grads);

            // push the gradient, this task is done only if the push is complete
            SArray<int> len = {};
            store_->Push(batch.feaids,
                         Store::kGradient,
                         grads,
                         lengths ? *lengths : len,
                         [this, on_complete]() { on_complete(); });
          } else {
            // a validation/prediction job
            on_complete();
          }
          if (values) delete values;
          if (lengths) delete lengths;
        };
        // pull the weight back
        store_->Pull(batch.feaids, Store::kWeight, values, lengths, pull_callback);
      });

  Reader* reader = nullptr;
  bool push_cnt = job.type == sgd::Job::kTraining && job.epoch == 0 && do_embedding_;
  push_cnt=false;

  if (job.type == sgd::Job::kTraining) {
    reader = new BatchReader(param_.data_in,
                             param_.data_format,
                             job.part_idx,
                             job.num_parts,
                             param_.batch_size,
                             param_.batch_size * param_.shuffle,
                             param_.neg_sampling);
  } else {
    reader = new Reader(param_.data_val,
                        param_.data_format,
                        job.part_idx,
                        job.num_parts,
                        256*1024*1024);
  }
  while (reader->Next()) {
    // map feature id into continous index
    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaids = std::make_shared<std::vector<feaid_t>>();
    auto feacnt = std::make_shared<std::vector<real_t>>();
    Localizer lc(-1, blk_nthreads_);
    lc.Compact(reader->Value(), data, feaids.get(), push_cnt ? feacnt.get() : nullptr);

    // save results into batch
    BatchJob batch;
    batch.type = job.type;
    batch.feaids = SArray<feaid_t>(feaids);
    batch.data = SharedRowBlockContainer<unsigned>(&data);
    delete data;

    // push feature count into the servers
    if (push_cnt) {
      store_->Wait(store_->Push(
          batch.feaids, Store::kFeaCount, SArray<real_t>(feacnt), {}));
    }

    // avoid too many batches are processing in parallel
    while (batch_tracker.NumRemains() > 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    batch_tracker.Issue({batch});
  }
  batch_tracker.Wait();
  delete reader;
}

real_t SGDLearner::EvaluatePenalty(const SArray<real_t>& weights,
                                   const SArray<int>& w_pos,
                                   const SArray<int>& V_pos) {
  real_t objv = 0;
  auto param = GetUpdater()->param();
  if (w_pos.size()) {
    for (int p : w_pos) {
      if (p == -1) continue;
      real_t w = weights[p];
      objv += param.l1 * fabs(w) + .5 * param.l2 * w * w;
    }
    for (int p : V_pos) {
      if (p == -1) continue;
      for (int i = 0; i < param.V_dim; ++i) {
        real_t V = weights[p+i];
        objv += .5 * param.V_l2 * V * V;
      }
    }
  } else {
    for (auto w : weights) {
      objv += param.l1 * fabs(w) + .5 * param.l2 * w * w;
    }
  }
  return objv;
}

}  // namespace difacto
