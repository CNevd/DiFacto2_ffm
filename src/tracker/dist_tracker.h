/**
 *  Copyright (c) 2016 by Contributors
 */
#ifndef DIFACTO_TRACKER_DIST_TRACKER_H_
#define DIFACTO_TRACKER_DIST_TRACKER_H_
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include "ps/ps.h"
#include "difacto/tracker.h"
#include "difacto/node_id.h"
#include "reader/workload_pool.h"
#include "sgd/sgd_utils.h"

namespace difacto {

static const int kSendWorkload = 1;
static const int kStopExec = 2;
using Callback = std::function<void()>;

/**
 * \brief a tracker which runs over mutliple machines
 */
class DistTracker : public Tracker {
 public:
  typedef std::pair<int, std::string> JobArgs;
  typedef std::string JobRets;
  DistTracker() {
    using namespace std::placeholders;
    app_ = new ps::SimpleApp(-1);
    // call JobHandle to process request from customer -1
    app_->set_request_handle(
        std::bind(&DistTracker::ReqHandle, this, _1, _2));
    app_->set_response_handle(
        std::bind(&DistTracker::RespHandle, this, _1, _2));
    monitor_thread_ = std::unique_ptr<std::thread>(new std::thread(&DistTracker::Monitoring, this));
  }
  ~DistTracker() {
    ps::Finalize();
    delete app_;
    monitor_thread_->join();
  }

  KWArgs Init(const KWArgs& kwargs) override {
    auto remain = pool_.Init(kwargs);
    return remain;
  }

  void Issue(const std::vector<std::pair<int, std::string>>& jobs) override { }

  /**
   * \brief send jobs to nodes and wait them finished.
   */
  void IssueAndWait(int node_id, std::string args) override {
    int ts = Send(-1, args, node_id);
    app_->Wait(ts);
  }

  void StartDispatch(int num_parts, int job_type, int epoch) {
    job_type_ = job_type;
    epoch_ = epoch;
    nparts_ = num_parts;
    pool_.Clear();
    pool_.Add(num_parts);
    // send an empty job to wake up workers
    Send(kSendWorkload, "", NodeID::kWorkerGroup);
  }

  /**
   * \brief set the async executor function worker/server
   */
  void SetExecutor(const Executor& executor) override {
    CHECK_NOTNULL(executor);
    executor_ = executor;
  }

  void SetMonitor(const Monitor& monitor) override {
    CHECK_NOTNULL(monitor);
    monitor_ = monitor;
  }

  int NumRemains() override {
    return pool_.NumRemains();
  }

  void Clear() override {
    pool_.ClearRemain();
  }

  /**
   * \block as a daemon until producer called Stop
   */
  void Wait() override {
    std::unique_lock<std::mutex> lk(mu_);
    run_cond_.wait(lk, [this] {return done_;});
  }

  void Stop() override {
    while(NumRemains()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    int ts = Send(kStopExec, "", NodeID::kServerGroup + NodeID::kWorkerGroup);
    app_->Wait(ts);
    done_ = true;
  }

 private:

  // for W is workload, for S is save/load model call executor to process
  // Or stop executor
  void ReqHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopExec) {
      done_ = true;
      run_cond_.notify_one();
      app_->Response(recved);
      return;
    }
    CHECK(executor_) << "set executor first";
    JobRets rets;
    if (!recved.body.empty()) {
      executor_(recved.body, &rets);
    }
    app_->Response(recved, rets);
  }

  // just for SCH to handle response form W and S (W: file process resp  S: save/load model response)
  // however resp from S just ignore
  void RespHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head != kSendWorkload) return;
    auto id = recved.sender;
    JobRets rets = recved.body;
    pool_.Finish(id);
    if (monitor_ && rets.size()) monitor_(id, rets);

    // send a new workload
    int k = pool_.Get(id);
    if (k < -1) return;

    std::string job_str;
    sgd::Job job;
    job.type = job_type_;
    job.epoch = epoch_;
    job.num_parts = nparts_;
    job.part_idx = k;
    job.SerializeToString(&job_str);

    Send(kSendWorkload, job_str, id);
  }

  inline int Send(int cmd_id,
           const std::string& cmd_body,
           const int receiver) {
    return app_->Request(cmd_id, cmd_body, receiver);
  }
  
  void Monitoring() {
    while (!done_) {
      auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(0);
      if (dead_nodes.size()) {
        if (IsScheduler()) {
          for (auto id : dead_nodes) {
            pool_.Reset(id);
          }
        } else {
          LOG(WARNING) << "Scheduler is died, Stop myself";
          ForceExit();
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  }

  void ForceExit() {
    std::string kill = "kill -9 " + std::to_string(getpid());
    int ret = system(kill.c_str());
    if (ret != 0) LOG(INFO) << "failed to " << kill;
  }

  bool done_ = false;
  int job_type_, epoch_ = 0, nparts_ = 0;
  std::mutex mu_;
  std::condition_variable run_cond_;
  Executor executor_;
  Monitor monitor_;
  ps::SimpleApp* app_;
  WorkloadPool pool_;
  std::unique_ptr<std::thread> monitor_thread_;
};
}  // namespace difacto
#endif  // DIFACTO_TRACKER_DIST_TRACKER_H_
