/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_REPORTER_DIST_REPORTER_H_
#define DIFACTO_REPORTER_DIST_REPORTER_H_
#include <string>
#include "ps/ps.h"
#include "difacto/node_id.h"

namespace difacto {

class DistReporter : public Reporter {
 public:
  DistReporter() {
    using namespace std::placeholders;
    app_ = new ps::SimpleApp(-2);
    // call reporter Handle to process request from customer -2
    app_->set_request_handle(
        std::bind(&DistReporter::ReqHandle, this, _1, _2));
    app_->set_response_handle(
        std::bind(&DistReporter::RespHandle, this, _1, _2));
  }
  virtual ~DistReporter() {
    delete app_;
  }

  KWArgs Init(const KWArgs& kwargs) override { return kwargs; }

  void SetMonitor(const Monitor& monitor) override {
    CHECK_NOTNULL(monitor);
    monitor_ = monitor;
  }

  // Report progress to Scheduler
  int Report(const std::string& report) override{
    int ts = Send(-1, report, NodeID::kScheduler);
    return ts;
  }

  void Wait(int timestamp) override{ app_->Wait(timestamp); }

 private:

 inline int Send(int cmd_id,
           const std::string& cmd_body,
           const int receiver) {
    return app_->Request(cmd_id, cmd_body, receiver);
  }

  void ReqHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    auto id = recved.sender;
    if (monitor_ && !recved.body.empty()) monitor_(id, recved.body);
  }

  void RespHandle(const ps::SimpleData& recved, ps::SimpleApp* app) { }

  Monitor monitor_;
  ps::SimpleApp* app_;
};
}  // namespace difacto
#endif  // DIFACTO_REPORTER_DIST_REPORTER_H_
