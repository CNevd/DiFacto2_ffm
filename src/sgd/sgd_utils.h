/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_UTILS_H_
#define DIFACTO_SGD_SGD_UTILS_H_
#include <string>
#include <vector>
#include <sstream>
#include "dmlc/memory_io.h"
namespace difacto {
namespace sgd {

/**
 * \brief a sgd job
 */
struct Job {
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kTraining = 3;
  static const int kValidation = 4;
  static const int kPrediction = 5;
  static const int kEvaluation = 6;
  /** \brief job type */
  int type;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;
  Job() { }
  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(type);
    ss->Write(num_parts);
    ss->Write(part_idx);
    ss->Write(epoch);
    delete ss;
  }

  void ParseFromString(const std::string& str) {
    auto copy = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&copy);
    ss->Read(&type);
    ss->Read(&num_parts);
    ss->Read(&part_idx);
    ss->Read(&epoch);
    delete ss;
  }
};

struct Progress {
  real_t nrows = 0;  // number of examples
  real_t loss = 0;  //
  real_t auc = 0;   // auc
  real_t penalty = 0;  //
  real_t nnz_w = 0;  // |w|_0

  std::string TextString() {
    std::stringstream ss;
    ss <<"Rows = " << nrows << ", loss = " << loss / nrows << ", AUC = " << auc / nrows;
    return ss.str();
  }

  void SerializeToString(std::string* str) const {
    *str = std::string(reinterpret_cast<char const*>(this), sizeof(Progress));
  }

  void ParseFrom(char const* data, size_t size) {
    if (size == 0) return;
    CHECK_EQ(size, sizeof(Progress));
    memcpy(this, data, sizeof(Progress));
  }

  void Merge(const std::string& str) {
    Progress other;
    other.ParseFrom(str.data(), str.size());
    Merge(other);
  }

  void Merge(const Progress& other) {
    size_t n = sizeof(Progress) / sizeof(real_t);
    auto a = reinterpret_cast<real_t*>(this);
    auto b = reinterpret_cast<real_t const*>(&other);
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
  }

  void Reset() {
    loss = 0; penalty = 0;
    auc = 0; nnz_w = 0;
    nrows = 0;
  }
};

struct Report_prog {
  Progress prog;
  real_t nrows = 0;
  real_t nnz_w = 0;

  std::string PrintStr() {
    nrows += prog.nrows;
    nnz_w += prog.nnz_w;

    char buf[256];
    snprintf(buf, 256, "%9.4g  %7.2g | %9.4g | %6.4lf  %7.5lf ",
             nrows, prog.nrows, nnz_w, prog.loss / prog.nrows, prog.auc / prog.nrows);
    prog.Reset(); 
    return std::string(buf);
  }
};

}  // namespace sgd
}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_UTILS_H_
