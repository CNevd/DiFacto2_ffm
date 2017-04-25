/**
 * Copyright (c) 2016 by Contributors
 * @file   store_dist.h
 * @brief  distributed implementation based on ps-lite
 */

#ifndef DIFACTO_STORE_KVSTORE_DIST_H_
#define DIFACTO_STORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include <functional>
#include "ps/ps.h"
#include "difacto/store.h"
#include "difacto/updater.h"
#include "dmlc/parameter.h"
#include "./kvstore_dist_server.h"
namespace difacto {

/*! \brief kvstore parameters for distributed model*/
struct KVStoreParam : public dmlc::Parameter<KVStoreParam> {
  std::string type;
  int delay;
  DMLC_DECLARE_PARAMETER(KVStoreParam) {
    // declare parameters
    DMLC_DECLARE_FIELD(type).set_default("async")
          .describe("can be async,sync,ssp.");
    DMLC_DECLARE_FIELD(delay).set_range(1, 50).set_default(4)
          .describe("Bounded Delay for ssp model");
  }
};

/**
 * \brief distributed kvstore
 *
 * for a worker node, it always guarantees that all push and pull issued from
 * this worker on the same key are serialized. namely push(3) and then pull(3),
 * then the data pulled is always containing the modification from the push(3).
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public Store {
 public:
  KVStoreDist() : ps_worker_(nullptr), server_(nullptr) { 
    if (IsWorker()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("difacto_worker\0");
    } else {
      if (IsServer()) {
        server_ = new KVStoreDistServer();
      }
      ps::StartAsync("difacto_server\0");
    }

    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
  }

  virtual ~KVStoreDist() {
   if (IsWorker()) {
     if (barrier_before_exit_) Barrier();
   }
   if (ps_worker_) {delete ps_worker_; ps_worker_ = nullptr;}
   if (server_) {delete server_; server_ = nullptr;}
  }

  KWArgs Init(const KWArgs& kwargs) {
    auto remain = kvparam.InitAllowUnknown(kwargs);
    return remain;
  }

  int Push(const SArray<feaid_t>& fea_ids,
           int val_type,
           const SArray<real_t>& vals,
           const SArray<int>& lens,
           const std::function<void()>& on_complete) override {
    CHECK(IsKeysOrderd(fea_ids)) << "fea_ids must in non-decreasing order";
    return CHECK_NOTNULL(ps_worker_)->ZPush(
	       fea_ids, vals, lens, val_type, on_complete);
  }

  int Pull(const SArray<feaid_t>& fea_ids,
           int val_type,
           SArray<real_t>* vals,
           SArray<int>* lens,
           const std::function<void()>& on_complete) override {
    CHECK(IsKeysOrderd(fea_ids)) << "fea_ids must in non-decreasing order";
    return CHECK_NOTNULL(ps_worker_)->ZPull(
           fea_ids, vals, lens, val_type, on_complete);
  }

  /** \brief set an updater, only required for a server node */
  void SetUpdater(const std::shared_ptr<Updater>& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServer()) {
      CHECK_NOTNULL(server_)->SetUpdater(updater);
    }
    updater_ = updater;
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }

  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) { 
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  // wait until task-time is finished
  void Wait(int time) override {
    CHECK_NOTNULL(ps_worker_)->Wait(time);
  }
  int NumWorkers() override { return ps::NumWorkers(); }
  int NumServers() override { return ps::NumServers(); }
  int Rank() override { return ps::MyRank(); }

 private:
  inline bool IsKeysOrderd(const SArray<feaid_t>& keys) {
    for (size_t i = 0; i < keys.size()-1; ++i) {
      if (keys[i+1] < keys[i]) {return false;}
    }
    return true;
  }

  KVStoreParam kvparam;

  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
};

}  // namespace difacto


#endif  // DIFACTO_STORE_KVSTORE_DIST_H_
