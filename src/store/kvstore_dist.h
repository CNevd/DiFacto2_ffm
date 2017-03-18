/**
 * Copyright (c) 2016 by Contributors
 * @file   kvstore_dist.h
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
#include "common/threadsafe_queue.h"
#include "dmlc/parameter.h"
#include "vector_clock.h"
namespace difacto {

/*! \brief kvstore parameters for distributed model*/
struct KVStoreParam : public dmlc::Parameter<KVStoreParam> {
  bool sync_mode;
  int max_delay;
  DMLC_DECLARE_PARAMETER(KVStoreParam) {
    // declare parameters
    DMLC_DECLARE_FIELD(sync_mode).set_default(false)
          .describe("false for async, true for sync.");
    DMLC_DECLARE_FIELD(max_delay).set_range(0, 99).set_default(0)
          .describe("Bounded Delay for sync model.");
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
  KVStoreDist() : ps_worker_(nullptr), ps_server_(nullptr) { 
    if (IsWorker()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("difacto_worker\0");
    } else {
      if (IsServer()) {
        using namespace std::placeholders;
        ps_server_ = new ps::KVServer<float>(0);
        static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
            std::bind(&KVStoreDist::CommandHandle, this, _1, _2));
        ps_server_->set_request_handle(
            std::bind(&KVStoreDist::DataHandle, this, _1, _2, _3));
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
    delete ps_worker_;
    delete ps_server_;
  }

  KWArgs Init(const KWArgs& kwargs) {
    auto remain = kvparam.InitAllowUnknown(kwargs);
    /*! vector clock */
    if (kvparam.sync_mode) {
        worker_pull_clocks_.reset(new VectorClock(ps::NumWorkers()));
        worker_push_clocks_.reset(new VectorClock(ps::NumWorkers()));
        num_waited_push_.resize(ps::NumWorkers(), 0);
    }
    return remain;
  }


  /*!
   * \brief functions for a worker
   * */
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

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }

  /*! \brief wait until task-time is finished */
  void Wait(int time) override { CHECK_NOTNULL(ps_worker_)->Wait(time); }
  int NumWorkers() override { return ps::NumWorkers(); }
  int NumServers() override { return ps::NumServers(); }
  int Rank() override { return ps::MyRank(); }


  /*!
   * \ brief data handle for a server
   */
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    LOG(FATAL) << "TODO";
    app->Response(recved);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
    CHECK_GT(req_data.keys.size(), (size_t)0) << "req_data must has keys";
    CHECK(updater_);
    int sender_rank = ps::Postoffice::Get()->IDtoRank(req_meta.sender);
    if (req_meta.push) {
      CHECK_GT(req_data.vals.size(), (size_t)0) << "pushed req_data must has vals";
 
      if (kvparam.sync_mode) {
        // synced push SSP BSP
        if (worker_pull_clocks_->local_clock(sender_rank) >
            worker_pull_clocks_->global_clock()) {
          MsgBuf msg;
          msg.data.keys.CopyFrom(req_data.keys);
          msg.data.vals.CopyFrom(req_data.vals);
          msg.request = req_meta;
          msg_push_buf_.Push(msg);
          ++num_waited_push_[sender_rank];
          return;
        }
        // process push
        HandlePush(req_meta, req_data, server);
        if (worker_push_clocks_->Update(sender_rank)) {
          CHECK(msg_push_buf_.Empty());
          while (!msg_pull_buf_.Empty()) {
            MsgBuf msg;
            CHECK(msg_pull_buf_.TryPop(msg));
            HandlePull(msg.request, msg.data, server);
            int rank = ps::Postoffice::Get()->IDtoRank(msg.request.sender);
            CHECK(!worker_pull_clocks_->Update(rank));
          }
        }
      } else {
        // async push
        HandlePush(req_meta, req_data, server);
      }
    } else {
      if (kvparam.sync_mode) {
        // synced pull SSP BSP
        if (worker_push_clocks_->local_clock(sender_rank) >
            worker_push_clocks_->global_clock() ||
            num_waited_push_[sender_rank] > 0) {
          MsgBuf msg;
          msg.data.keys.CopyFrom(req_data.keys);
          msg.data.vals.CopyFrom(req_data.vals);
          msg.request = req_meta;
          msg_pull_buf_.Push(msg);
          return;
        }
        HandlePull(req_meta, req_data, server);
        if (worker_pull_clocks_->Update(sender_rank)) {
          while (!msg_push_buf_.Empty()) {
            MsgBuf msg;
            CHECK(msg_push_buf_.TryPop(msg));
            HandlePush(msg.request, msg.data, server);
            int rank = ps::Postoffice::Get()->IDtoRank(msg.request.sender);
            CHECK(!worker_push_clocks_->Update(rank));
            --num_waited_push_[rank];
          }
        }
      } else {
        // async pull
        HandlePull(req_meta, req_data, server);
      }
    }
  }

 
 private:

  void HandlePush (const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
    int val_type = req_meta.cmd;
    updater_->Update(req_data.keys, val_type, req_data.vals, req_data.lens);
    server->Response(req_meta);
    Report();
  }

  void HandlePull (const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
    int val_type = req_meta.cmd;
    ps::KVPairs<real_t> response;
    updater_->Get(req_data.keys, val_type, &(response.vals), &(response.lens));
    response.keys = req_data.keys;
    server->Response(req_meta, response);
  }

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
  ps::KVServer<real_t>* ps_server_;

  /**
   * \brief vector clock for server
   */ 
  std::unique_ptr<VectorClock> worker_pull_clocks_;
  std::unique_ptr<VectorClock> worker_push_clocks_;
  std::vector<int> num_waited_push_;

  struct MsgBuf {
    ps::KVMeta request;
    ps::KVPairs<real_t> data;
  };

  ThreadsafeQueue<MsgBuf> msg_push_buf_;
  ThreadsafeQueue<MsgBuf> msg_pull_buf_;
};

}  // namespace difacto


#endif  // DIFACTO_STORE_KVSTORE_DIST_H_
