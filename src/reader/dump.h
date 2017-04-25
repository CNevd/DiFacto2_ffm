/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_READER_DUMP_H_
#define DIFACTO_READER_DUMP_H_
#include <string>
#include "dmlc/parameter.h"
#include "dmlc/io.h"
#include "sgd/sgd_updater.h"
namespace difacto {

struct DumpParam : public dmlc::Parameter<DumpParam> {
  /** \brief name of updater */
  std::string updater;
  /** \brief the model file to dump */
  std::string model_in;
  /** \brief the dump file name */
  std::string name_dump;
  /** \brief wether to reverse the feature id */
  bool need_reverse;
  /** \brief wether dump aux data */
  bool dump_aux;
  
  DMLC_DECLARE_PARAMETER(DumpParam) {
    DMLC_DECLARE_FIELD(updater).set_default("sgd");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(name_dump).set_default("dump.txt");
    DMLC_DECLARE_FIELD(need_reverse).set_default(false);
    DMLC_DECLARE_FIELD(dump_aux).set_default(false);
  };
};
/**
 * \brief model dumper
 */
class Dump {
 public:
  KWArgs Init(const KWArgs& kwargs) {
    auto remain = param_.InitAllowUnknown(kwargs);
    return remain;
  }

  void Run() {
    // load model_in fisrt
    if (!param_.model_in.size()) {
      LOG(FATAL) << "Pls set model_in";
      return;
    }

    if (param_.updater == "sgd"){
      updater_ = std::shared_ptr<Updater>(new SGDUpdater());
    } else {
      LOG(FATAL) << "Unkonwn updater: " << param_.updater;
    }

    std::unique_ptr<dmlc::Stream> fi(
            dmlc::Stream::Create(param_.model_in.c_str(), "r"));
    updater_->Load(fi.get());

    // dump model
    std::unique_ptr<dmlc::Stream> fo(
            dmlc::Stream::Create(param_.name_dump.c_str(), "w"));
    updater_->Dump(param_.dump_aux, param_.need_reverse, fo.get());
  }

 private:
  DumpParam param_;
  std::shared_ptr<Updater> updater_;
};

}  // namespace difacto

#endif  // DIFACTO_READER_DUMP_H_
