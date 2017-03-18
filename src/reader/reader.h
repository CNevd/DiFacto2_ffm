/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_READER_READER_H_
#define DIFACTO_READER_READER_H_
#include <string>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "data/parser.h"
#include "data/libfm_parser.h"
namespace difacto {
/**
 * \brief a reader reads a chunk of data with roughly same size a time
 */
class Reader {
 public:
  Reader() { parser_ = nullptr; }
  Reader(const std::string& uri,
         const std::string& format,
         int part_index,
         int num_parts,
         int chunk_size_hint) {
    char const* c_uri = uri.c_str();
    dmlc::InputSplit* input = dmlc::InputSplit::Create(
        c_uri, part_index, num_parts, format == "rec" ? "recordio" : "text");
    input->HintChunkSize(chunk_size_hint);

    if (format == "libfm") {
      parser_ = new dmlc::data::LibFMParser<feaid_t>(input, 1);
    } else {
      LOG(FATAL) << "unknown format " << format;
    }
    parser_ = new dmlc::data::ThreadedParser<feaid_t>(parser_);
  }

  virtual ~Reader() { delete parser_; }

  virtual bool Next() { return parser_->Next(); }

  virtual const dmlc::RowBlock<feaid_t>& Value() const { return parser_->Value(); }

 private:
  dmlc::data::ParserImpl<feaid_t>* parser_;
};

}  // namespace difacto
#endif  // DIFACTO_READER_READER_H_
