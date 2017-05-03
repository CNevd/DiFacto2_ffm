/**
 * Copyright (c) 2015 by Contributors
 * @file   sgd.h
 * @brief  the stochastic gradient descent solver
 */
#ifndef DIFACTO_SGD_SGD_UPDATER_H_
#define DIFACTO_SGD_SGD_UPDATER_H_
#include <vector>
#include <mutex>
#include <limits>
#include <random>
#include "dmlc/io.h"
#include "difacto/updater.h"
#include "./sgd_param.h"
#include "./sgd_utils.h"
namespace difacto {

/**
 * \brief the weight entry for one feature
 */
struct SGDEntry {
 public:
  SGDEntry() { }
  ~SGDEntry() { delete [] V; delete [] Z;}
  /** \brief the number of appearence of this feature in the data so far */
  real_t fea_cnt = 0;
  /** \brief V and its aux data */
  real_t *V = nullptr;
  real_t *Z = nullptr; // cg and z
  /** \brief size of V */
  int size = 0;
  int nnz = 0;
  /** \brief wether entry is empty */
  inline bool empty() const { return nnz == 0; }
  /** \brief save this entry */
  void SaveEntry(bool save_aux, dmlc::Stream* fo) const {
    if (size == 0) return;
    fo->Write(&size, sizeof(size));
    // save V
    fo->Write(V, sizeof(real_t)*size);
    if (save_aux) fo->Write(Z, sizeof(real_t)*size*2);
  }
  /** \brief load this entry */
  void LoadEntry(dmlc::Stream* fi, bool has_aux) {
    CHECK_EQ(fi->Read(&size, sizeof(size)), sizeof(size));
    // load V
    V = new real_t[size];
    Z = new real_t[size * 2]; // cg and z
    CHECK_EQ(fi->Read(V, sizeof(real_t)*size), sizeof(real_t)*size);
    for (int i = 0; i < size; ++i) { if (V[i] != 0) nnz += 1; }
    if (has_aux) {
      CHECK_EQ(fi->Read(Z, sizeof(real_t)*size*2), sizeof(real_t)*size*2);
    } else {
      memset(Z, 0, sizeof(real_t)*size*2);
    }
  }
};
/**
 * \brief sgd updater
 *
 * - w is updated by FTRL, which is a smooth version of adagrad works well with
 *   the l1 regularizer
 * - V is updated by adagrad
 */
class SGDUpdater : public Updater {
 public:
  SGDUpdater() {}
  virtual ~SGDUpdater() {}

  KWArgs Init(const KWArgs& kwargs) override;

  void Load(dmlc::Stream* fi) override {
    bool has_aux;
    feaid_t key;
    int64_t loaded = 0;
    if (fi->Read(&has_aux, sizeof(bool)) != sizeof(bool)) return;
    while (true) {
      if (fi->Read(&key, sizeof(feaid_t)) != sizeof(feaid_t)) break;
      model_[key].LoadEntry(fi, has_aux);
      loaded ++ ;
      new_w += model_[key].nnz;
    }
    LOG(INFO) << "loaded " << loaded << " kv pairs";
  };

  void Save(bool save_aux, dmlc::Stream *fo) const override {
    int64_t saved = 0;
    fo->Write(&save_aux, sizeof(bool));
    for (const auto& it : model_) {
      if (it.second.empty()) continue;
      fo->Write(&it.first, sizeof(feaid_t));
      it.second.SaveEntry(save_aux, fo);
      saved ++ ;
    }
    LOG(INFO) << "saved " << saved << " kv pairs";
  };

  void Dump(bool dump_aux, bool need_reverse, dmlc::Stream *fo) const override {
    int64_t dumped = 0;
    dmlc::ostream os(fo);
    for (const auto& it : model_) {
      if (it.second.empty()) continue;
      auto key = need_reverse ? ReverseBytes(it.first) : it.first;
      os << key;

      /** \ here dump entry to avoid passing os */
      os << '\t' << it.second.size;
      // dump V
      int n = it.second.size;
      for (int i = 0; i < n; ++i) {
        os << '\t' << it.second.V[i];
      }
      if (dump_aux) {
        for (int i = 0; i < n*2; ++i) {
          os << '\t' << it.second.Z[i];
        }
      }

      os << '\n';
      dumped ++ ;
    }
    LOG(INFO) << "dumped " << dumped << " kv pairs";
  };

  std::string Get_report() override {
    sgd::Progress report_prog; report_prog.nnz_w = new_w;
    std::string rets;
    report_prog.SerializeToString(&rets);
    new_w = 0;
    return rets;
  };
  
  void Get(const SArray<feaid_t>& fea_ids,
           int value_type,
           SArray<real_t>* weights,
           SArray<int>* val_lens) override;


  void Update(const SArray<feaid_t>& fea_ids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& val_lens) override;

  void Evaluate(sgd::Progress* prog) const;

  const SGDUpdaterParam& param() const { return param_; }

 private:

  /** \brief update V by adagrad */
  void UpdateV(real_t const* gV, SGDEntry* e);

  /** \brief init V */
  void InitV(SGDEntry* e);

  /** \brief new w for a server */
  float new_w = 0;

  /** \brief dim of a feature */
  int feat_dim = 0;

  /** \brief coef for model initialization*/
  float coef = 1.0;

  /** \brief generator for model initialization */
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution;

  SGDUpdaterParam param_;
  std::unordered_map<feaid_t, SGDEntry> model_;
  mutable std::mutex mu_;
};


}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_UPDATER_H_
