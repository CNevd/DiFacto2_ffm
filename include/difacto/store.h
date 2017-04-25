/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_STORE_H_
#define DIFACTO_STORE_H_
#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include "./base.h"
#include "dmlc/io.h"
#include "dmlc/parameter.h"
#include "./sarray.h"
#include "./updater.h"
#include "./reporter.h"
namespace difacto {

/**
 * \brief the store allows workers to get and set and model
 */
class Store {
 public:
  /**
   * \brief the factory function
   */
  static Store* Create();

  /** \brief default constructor */
  Store() { }
  /** \brief default deconstructor */
  virtual ~Store() { }

  static const int kFeaCount = 1;
  static const int kWeight = 2;
  static const int kGradient = 3;
  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief push a list of (feature id, value) into the store
   *
   * @param sync_type
   * @param fea_ids
   * @param vals
   * @param lens
   * @param on_complete
   *
   * @return timestamp
   */
  virtual int Push(const SArray<feaid_t>& fea_ids,
                   int val_type,
                   const SArray<real_t>& vals,
                   const SArray<int>& lens,
                   const std::function<void()>& on_complete = nullptr) = 0;
  /**
   * \brief pull the values for a list of feature ids
   *
   * @param sync_type
   * @param fea_ids
   * @param vals
   * @param lens
   * @param on_complete
   *
   * @return timestamp
   */
  virtual int Pull(const SArray<feaid_t>& fea_ids,
                   int val_type,
                   SArray<real_t>* vals,
                   SArray<int>* lens,
                   const std::function<void()>& on_complete = nullptr) = 0;


  /**
   * \brief wait until a push or a pull is actually finished
   *
   * @param time
   */
  virtual void Wait(int time) = 0;

  /**
   * \brief return number of workers
   */
  virtual int NumWorkers() = 0;
  /**
   * \brief return number of servers
   */
  virtual int NumServers() = 0;
  /**
   * \brief return the rank of this node
   */
  virtual int Rank() = 0;
  /**
   * \brief set an updater for the store, only required for a server node
   */
  virtual void SetUpdater(const std::shared_ptr<Updater>& updater) {
    CHECK(updater);
    updater_ = updater;
  }
  /**
   * \brief get the updater
  */
  std::shared_ptr<Updater> updater() { return updater_; }
  /** 
   * \brief set the reporter function only for a server node
   */
  void SetReporter(const std::shared_ptr<Reporter>& reporter) {
    CHECK(reporter);
    reporter_ = reporter;
  };
  /**
   * \brief default reporter to the scheduler for a server node
   */
  inline void Report() {
    if (reporter_ && updater_ && ++ct_ > 50) {
      reporter_->Report(updater_->Get_report());
      ct_ = 0;
    }
  }

  /******************************************************
   * the following are used for multi-machines.
   ******************************************************/

  /** \brief set whether to do barrier when finalize only for W */
  virtual void set_barrier_before_exit(const bool barrier_before_exit) {
    barrier_before_exit_ = barrier_before_exit;
  }

  /*!
   * \brief global barrier among all worker machines
   *
   * But note that, this functions only blocks the main thread of workers until
   * all of them are reached this point. It doesn't guarantee that all
   * operations issued before are actually finished, such as \ref Push and \ref Pull.
   */
  virtual void Barrier() { }

  virtual void RunServer() { }


 protected:
  /**
   * \brief the user-defined  updater
   */
  std::shared_ptr<Updater> updater_;

  /**
   * \brief the reporter function
   */
  std::shared_ptr<Reporter> reporter_;

  /**
   * \brief whether to do barrier when finalize
   */
  std::atomic<bool> barrier_before_exit_{true};

  int ct_ = 0;
};

}  // namespace difacto

#endif  // DIFACTO_STORE_H_
