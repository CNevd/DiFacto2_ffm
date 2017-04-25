/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_TRACKER_LOCAL_TRACKER_H_
#define DIFACTO_TRACKER_LOCAL_TRACKER_H_
#include <vector>
#include <utility>
#include <string>
#include "difacto/tracker.h"
#include "difacto/node_id.h"
#include "sgd/sgd_utils.h"
#include "./async_local_tracker.h"

namespace difacto {
/**
 * \brief an implementation of the tracker which only runs within a local
 * process
 */
class LocalTracker : public Tracker {
 public:
  typedef std::pair<int, std::string> Job;

  LocalTracker() {
    tracker_ = new AsyncLocalTracker<Job, Job>();
  }
  virtual ~LocalTracker() { delete tracker_; }

  KWArgs Init(const KWArgs& kwargs) override { return kwargs; }


  void Issue(const std::vector<Job>& jobs) override {
    if (!tracker_) tracker_ = new AsyncLocalTracker<Job, Job>();
    tracker_->Issue(jobs);
  }

  void IssueAndWait(int node_id, std::string args) override {
    Issue({std::make_pair(node_id, args)});
    tracker_->Wait(0);
  };

  void StartDispatch(int num_parts, int job_type, int epoch) {
    std::vector<std::pair<int, std::string>> jobs(num_parts);
    for (int i = 0; i < num_parts; ++i) {
      jobs[i].first = NodeID::kWorkerGroup;
      sgd::Job job;
      job.type = job_type;
      job.epoch = epoch;
      job.num_parts = num_parts;
      job.part_idx = i;
      job.SerializeToString(&jobs[i].second);
    }
    Issue(jobs);
  };

  int NumRemains() override {
    return CHECK_NOTNULL(tracker_)->NumRemains();
  }

  void Clear() override {
    CHECK_NOTNULL(tracker_)->Clear();
  }

  void Stop() override {
    if (tracker_) {
      delete tracker_;
      tracker_ = nullptr;
    }
  }

  void Wait() override {
    CHECK_NOTNULL(tracker_)->Wait();
  }

  void SetMonitor(const Monitor& monitor) override {
    CHECK_NOTNULL(tracker_)->SetMonitor(
        [monitor](const Job& rets) {
          if (monitor) monitor(rets.first, rets.second);
        });
  }

  void SetExecutor(const Executor& executor) override {
    CHECK_NOTNULL(executor);
    CHECK_NOTNULL(tracker_)->SetExecutor(
        [executor](const Job& args,
                   const std::function<void()>& on_complete,
                   Job* rets) {
          rets->first = args.first;
          executor(args.second, &(rets->second));
          on_complete();
        });
  }

 private:
  AsyncLocalTracker<Job, Job>* tracker_ = nullptr;
};

}  // namespace difacto
#endif  // DIFACTO_TRACKER_LOCAL_TRACKER_H_
