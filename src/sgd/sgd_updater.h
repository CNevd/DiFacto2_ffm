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
  ~SGDEntry() { delete [] V; }
  /** \brief the number of appearence of this feature in the data so far */
  real_t fea_cnt = 0;
  /** \brief w and its aux data */
  real_t w = 0, sqrt_g = 0, z = 0;
  /** \brief V and its aux data */
  real_t *V = nullptr;
  /** \brief size of w+V */
  int size = 1;
  /** \brief wether entry is empty */
  inline bool empty() const { return (w == 0 && size == 1); }
  /** \brief save this entry */
  void SaveEntry(bool save_aux, dmlc::Stream* fo) const {
    fo->Write(&size, sizeof(size));
    // save w
    fo->Write(&w, sizeof(real_t));
    if (save_aux) {
      fo->Write(&sqrt_g, sizeof(real_t));
      fo->Write(&z, sizeof(real_t));
    }
    if (size == 1) return;
    // save V
    int n = size - 1;
    fo->Write(V, sizeof(real_t)*n);
    if (save_aux) fo->Write(V+n, sizeof(real_t)*n);
  }
  /** \brief load this entry */
  void LoadEntry(dmlc::Stream* fi, bool has_aux) {
    CHECK_EQ(fi->Read(&size, sizeof(size)), sizeof(size));
    // load w
    CHECK_EQ(fi->Read(&w, sizeof(real_t)), sizeof(real_t));
    if (has_aux) {
      CHECK_EQ(fi->Read(&sqrt_g, sizeof(real_t)), sizeof(real_t));
      CHECK_EQ(fi->Read(&z, sizeof(real_t)), sizeof(real_t));
    }
    if (size == 1) return;
    // load V
    int n = size - 1;
    V = new real_t[n*2];
    CHECK_EQ(fi->Read(V, sizeof(real_t)*n), sizeof(real_t)*n);
    if (has_aux) {
      CHECK_EQ(fi->Read(V+n, sizeof(real_t)*n), sizeof(real_t)*n);
    } else {
      memset(V+n, 0, sizeof(real_t)*n);
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
    if (fi->Read(&has_aux, sizeof(bool)) != sizeof(bool)) return;
    while (true) {
      if (fi->Read(&key, sizeof(feaid_t)) != sizeof(feaid_t)) break;
      model_[key].LoadEntry(fi, has_aux);
    }
    new_w = model_.size();
    LOG(INFO) << "loaded " << new_w << " kv pairs";
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
      // dump w
      os << '\t' << it.second.size << '\t' << it.second.w;
      if (dump_aux) {
        os << '\t' << it.second.sqrt_g << '\t' << it.second.z;
      }
      // dump V
      if (it.second.size > 1) {
        int n = it.second.size - 1;
        for (int i = 0; i < n; ++i) {
          os << '\t' << it.second.V[i];
        }
        if (dump_aux) {
          for (int i = n; i < 2*n; ++i) {
            os << '\t' << it.second.V[i];
          }
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
  /** \brief update w by FTRL */
  void UpdateW(real_t gw, SGDEntry* e);

  /** \brief update V by adagrad */
  void UpdateV(real_t const* gV, SGDEntry* e);

  /** \brief init V */
  void InitV(SGDEntry* e);

  /** \brief new w for a server */
  float new_w = 0;

  SGDUpdaterParam param_;
  std::unordered_map<feaid_t, SGDEntry> model_;
  mutable std::mutex mu_;
};


}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_UPDATER_H_
