/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_UPDATER_H_
#define DIFACTO_UPDATER_H_
#include <vector>
#include <string>
#include "./base.h"
#include "./sarray.h"
#include "dmlc/io.h"
namespace difacto {
/**
 * \brief the base class of an updater
 *
 * the main job of a updater is to update model based
 * on the received data (often gradient)
 */
class Updater {
 public:
  /**
   * \brief default constructor
   */
  Updater() { }
  /**
   * \brief default deconstructor
   */
  virtual ~Updater() { }
  /**
   * \brief init the updater
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief load the updater
   * \param fi input stream
   */
  virtual void Load(dmlc::Stream* fi) = 0;

  /**
   * \brief save the updater
   * \param save_aux whether or not save aux data
   * \param fo output stream
   */
  virtual void Save(bool save_aux, dmlc::Stream *fo) const = 0;

  /** 
   * \brief dump model to readable format
   * \param dump_aux whether or not dump aux data
   * \param need_reverse whether or not reverse feature id
   * \param fo output stream
   */
  virtual void Dump(bool dump_aux, bool need_reverse, dmlc::Stream *fo) const = 0;

  /**
   * \brief get the weights on the given features
   *
   * @param fea_ids the list of feature ids
   * @param model
   * @param model_offset could be empty
   */
  virtual void Get(const SArray<feaid_t>& fea_ids,
                   int data_type,
                   SArray<real_t>* data,
                   SArray<int>* data_offset) = 0;
  /**
   * \brief update the model given a list of key-value pairs
   *
   * @param fea_ids the list of feature ids
   * @param recv_data
   * @param recv_data_offset
   */
  virtual void Update(const SArray<feaid_t>& fea_ids,
                      int data_type,
                      const SArray<real_t>& data,
                      const SArray<int>& data_offset) = 0;

  /** \brief get report info of weight */
  virtual std::string Get_report() = 0;
 };

}  // namespace difacto
#endif  // DIFACTO_UPDATER_H_
