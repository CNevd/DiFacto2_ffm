#pragma once
#include <sstream>
#include <vector>
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include "dmlc/timer.h"
#include "dmlc/parameter.h"
namespace difacto {

/*! brief workload pool paramenters */
struct WorkloadPoolParam : public dmlc::Parameter<WorkloadPoolParam> {
  bool shuffle;
  double straggler_timeout;
  DMLC_DECLARE_PARAMETER(WorkloadPoolParam) {
    // declare parameters
    DMLC_DECLARE_FIELD(shuffle).set_default(true)
        .describe("whether use data shuffle in Workload pool");
    DMLC_DECLARE_FIELD(straggler_timeout).set_range(0, 99999).set_default(0)
        .describe("timeout for Straggler in msec");
  }
};

/**
 * @brief A thread-safe workload pool
 */
class WorkloadPool {
 public:
  WorkloadPool() { }
  ~WorkloadPool() {
    done_ = true;
    if (straggler_killer_) {
      straggler_killer_->join();
      delete straggler_killer_;
    }
  }

  KWArgs Init(const KWArgs& kwargs) {
    auto remain = wpparam.InitAllowUnknown(kwargs);
    if (wpparam.straggler_timeout) {
      straggler_killer_ = new std::thread([this]() {
        while (!done_) {
          // detecter straggler for every 2 second
          // make it longer in case there is server dead, need more time to recover
          // and process this workload
          RemoveStraggler(); sleep(2);
        }
      }); 
    }
    return remain;
  }

  void Add(int num_parts) {
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < num_parts; ++i) {
      track_[i] = 0;
    }
    CHECK_EQ(track_.size(), (size_t)num_parts);
    inited_ = true;
  }

  /*! \brief clear this workload pool */
  void Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    track_.clear();
    assigned_.clear();
    time_.clear();
    num_finished_ = 0;
    inited_ = false;
  }

  void ClearRemain() {
    std::lock_guard<std::mutex> lk(mu_);
    track_.clear();
  }

  /*!\brief get a part for a node */
  int Get(const int id) {
    std::lock_guard<std::mutex> lk(mu_);
    return GetOne(id);
  }

  /*! \brief reset workload if id dies */
  void Reset(const int id) { Set(id, false); }

  /*! \breif finish the workload this id got before */
  void Finish(const int id) { Set(id, true); }

  /*! \brief whether this pool inited */
  bool IsInited() { return inited_; }

  /*! \brief ramaining jobs */
  int NumRemains() {
    std::lock_guard<std::mutex> lk(mu_);
    return int(track_.size() + assigned_.size());
  }

 private:
  void Set(const int id, bool del) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      if (it->node == id) {
        if (!del) {
          track_[it->k] = 0;
          LOG(INFO) << id << " failed to finish workload " << it->DebugStr();
        } else {
          double time = dmlc::GetTime() - it->start;
          time_.push_back(time);
          auto it_t = track_.find(it->k);
          if (it_t != track_.end()) track_.erase(it_t);
          ++ num_finished_;
          LOG(INFO) << id << " finished " << it->DebugStr()
                    << " in " << time << " sec.";
        }
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }
  }

  int GetOne(const int id) {
    int pick = 0, i = 0;
    if (wpparam.shuffle) {
      int n = 0;
      for (auto& it : track_) {
        auto& t = it.second;
        if (t != 0) continue;
        ++ n;
      }
      if (n == 0) return -2;
      pick = rand() % n;
    }

    for (auto& it : track_) {
      auto& t = it.second;
      if (t != 0) continue;
      if (i < pick) { ++ i; continue; }
      // pick one part
      Assigned a;
      a.start = dmlc::GetTime();
      a.node  = id;
      a.k     = it.first;
      assigned_.push_back(a);
      LOG(INFO) << "assign " << id << " job " << a.DebugStr()
                << ". " << assigned_.size() << " #jobs on processing.";
      t = 1;
      return it.first;
    }
    return -2;
  }

  void RemoveStraggler() {
    std::lock_guard<std::mutex> lk(mu_);
    if (time_.size() < 10) return;
    double mean = 0;
    for (double t : time_) mean += t;
    mean /= time_.size();
    double cur_t = dmlc::GetTime();
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      double t = cur_t - it->start;
      if (t > std::max(mean * 10, wpparam.straggler_timeout)) {
        LOG(INFO) << it->node << " is processing "
                  << it->DebugStr() << " for " << t
                  << " sec, which is much longer than the average time "
                  << mean << " sec. reassign this workload to other nodes";
        track_[it->k] = 0;
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }
  }

  struct Assigned {
    int node;
    int k;
    double start;  // start time

    std::string DebugStr() {
      std::stringstream ss;
      ss << "Part: " << k;
      return ss.str();
    }
  };
  std::list<Assigned> assigned_;

  // state of each part
  // 0: available, 1: assigned, 2: done
  std::map<int, int> track_;
  int num_finished_ = 0;
  bool inited_ = false, done_ = false;

  WorkloadPoolParam wpparam;

  // process time of finished tasks
  std::vector<double> time_;
  std::mutex mu_;
  std::thread* straggler_killer_ = nullptr;
};

}  // namespace difacto
