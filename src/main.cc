/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "common/arg_parser.h"
#include "dmlc/parameter.h"
#include "reader/converter.h"
#include "reader/dump.h"
namespace difacto {

enum DifactoTask {
  kTrain = 0,
  kDumpModel = 1,
  kPredict = 2,
  kConvert = 3
};

struct DifactoParam : public dmlc::Parameter<DifactoParam> {
  /**
   * \brief the type of task,
   * - train: train a model, which is the default
   * - dump: dump model to readable format
   * - predict: predict by using a trained model
   * - convert: convert data from one format into another
   */
  int task;
  /** \brief the learner's type, required for a training task */
  std::string learner;
  DMLC_DECLARE_PARAMETER(DifactoParam) {
    DMLC_DECLARE_FIELD(learner).set_default("sgd");
    DMLC_DECLARE_FIELD(task).set_default(kTrain)
        .add_enum("train", kTrain)
        .add_enum("dump", kDumpModel)
        .add_enum("pred", kPredict)
        .add_enum("convert", kConvert)
        .describe("Task to be performed by the main program");
  }
};

void WarnUnknownKWArgs(const DifactoParam& param, const KWArgs& remain) {
  if (remain.empty()) return;
  LOG(WARNING) << "Unrecognized keyword argument for task = " << param.task;
  for (auto kw : remain) {
    LOG(WARNING) << " - " << kw.first << " = " << kw.second;
  }
}

DMLC_REGISTER_PARAMETER(DifactoParam);
DMLC_REGISTER_PARAMETER(ConverterParam);
DMLC_REGISTER_PARAMETER(DumpParam);

}  // namespace difacto

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto config_file key1=val1 key2=val2 ...";
    return 0;
  }
  using namespace difacto;

  // parse configuure
  ArgParser parser;
  parser.AddArgFile(argv[1]);
  for (int i = 2; i < argc; ++i) parser.AddArg(argv[i]);
  DifactoParam param;
  auto kwargs_remain = param.InitAllowUnknown(parser.GetKWArgs());

  // run task
  switch (param.task) {
    case kTrain:
      {
      Learner* learner = Learner::Create(param.learner);
      WarnUnknownKWArgs(param, learner->Init(kwargs_remain));
      LOG(INFO) << "start run learner";
      learner->Run();
      delete learner;
      }
      break;
    case kDumpModel:
      {
      Dump dumper;
      WarnUnknownKWArgs(param, dumper.Init(kwargs_remain));
      dumper.Run();
      }
      break;
    case kPredict:
      LOG(FATAL) << "TODO";
      break;
    case kConvert:
      {
      Converter converter;
      WarnUnknownKWArgs(param, converter.Init(kwargs_remain));
      converter.Run();
      }
      break;
    default:
      LOG(FATAL) << "unknown task: " << param.task;
      break;
  }

  return 0;
}
