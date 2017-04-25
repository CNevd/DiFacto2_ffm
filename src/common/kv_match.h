/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_KV_MATCH_H_
#define DIFACTO_COMMON_KV_MATCH_H_
#include <vector>
#include <thread>
#include <algorithm>
#include "dmlc/logging.h"
#include "./range.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
namespace difacto {
/**
 * \brief assignment operator
 */
enum AssignOp {
  ASSIGN,  // a = b
  PLUS,    // a += b
  MINUS,   // a -= b
  TIMES,   // a *= b
  DIVIDE,  // a -= b
  AND,     // a &= b
  OR,      // a |= b
  XOR      // a ^= b
};
/**
 * \brief return an assignment function: right op= left
 */
template<typename T>
inline void AssignFunc(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN: *rhs = lhs; break;
    case PLUS: *rhs += lhs; break;
    case MINUS: *rhs -= lhs; break;
    case TIMES: *rhs *= lhs; break;
    case DIVIDE: *rhs /= lhs; break;
    default: LOG(FATAL) << "use AssignOpInt..";
  }
}
}  // namespace difacto

/** \brief implementation */
#include "./kv_match-inl.h"

namespace difacto {
/**
 * \brief Merge \a src_val into \a dst_val by matching keys. Keys must be unique
 * and sorted, and value lenghths are fixed.
 *
 * \code
 * if (dst_key[i] == src_key[j]) {
 *    dst_val[i] op= src_val[j]
 * }
 * \endcode
 *
 * \code
 * src_key = {1,2,3};
 * src_val = {6,7,8};
 * dst_key = {1,3,5};
 * KVMatch(src_key, src_val, dst_key, &dst_val);
 * // then dst_val = {6,8,0};
 * \endcode
 * When finished, \a dst_val will have length `k * dst_key.size()` and filled
 * with matched value. Umatched value will be untouched if exists or filled with 0.
 *
 * \tparam K type of key
 * \tparam V type of value
 * \param src_key the source keys
 * \param src_val the source values
 * \param dst_key the destination keys
 * \param dst_val the destination values.
 * \param op the assignment operator (default is ASSIGN)
 * \param nthreads number of thread (default is 2)
 * \return the number of matched values
 */
template <typename K, typename V>
size_t KVMatch(
    const SArray<K>& src_key,
    const SArray<V>& src_val,
    const SArray<K>& dst_key,
    SArray<V>* dst_val,
    AssignOp op = ASSIGN,
    int nthreads = DEFAULT_NTHREADS) {
  // do check
  if (src_key.empty() || src_key.empty()) return 0;
  CHECK_GT(nthreads, 0);
  size_t val_len = src_val.size() / src_key.size();
  CHECK_EQ(src_key.size() * val_len, src_val.size());
  CHECK_NOTNULL(dst_val)->resize(dst_key.size() * val_len);

  // shorten the matching range
  auto range = ps::FindRange(dst_key, src_key.front(), src_key.back()+1);
  size_t grainsize = std::max(range.size() / nthreads + 5,
                              static_cast<size_t>(1024*1024));
  size_t n = 0;
  KVMatch<K, V>(
      src_key.data(), src_key.data() + src_key.size(), src_val.data(),
      dst_key.data() + range.begin(), dst_key.data() + range.end(),
      dst_val->data() + range.begin() * val_len, val_len, op, grainsize, &n);
  return n;
}

/**
 * \brief merge with various length values
 *
 * if src_offset is empty, fallback to the previous fixed value length version
 *
 * \param src_key the source keys
 * \param src_val the source values
 * \param src_len the length of the i-th source value, can be empty
 * \param dst_key the destination keys
 * \param dst_val the destination values.
 * \param dst_len the length of the i-th desstination values, might be empty
 * \param op the assignment operator (default is ASSIGN)
 * \param nthreads number of thread (default is 2)
 * \return the number of matched kv pairs
 */
template <typename K, typename I, typename V>
size_t KVMatch(
    const SArray<K>& src_key,
    const SArray<V>& src_val,
    const SArray<I>& src_len,
    const SArray<K>& dst_key,
    SArray<V>* dst_val,
    SArray<I>* dst_len,
    AssignOp op = ASSIGN,
    int nthreads = DEFAULT_NTHREADS) {
  // fallback to the fixed value length version
  if (src_len.empty()) {
    if (dst_len) dst_len->clear();
    return KVMatch(src_key, src_val, dst_key, dst_val, op, nthreads);
  }

  // do check
  CHECK_EQ(src_key.size(), src_len.size());

  // match length
  dst_len->clear();
  dst_len->resize(dst_key.size(), 0);
  KVMatch(src_key, src_len, dst_key, dst_len, ASSIGN, nthreads);

  // match value
  size_t size = 0;
  for (I i : *dst_len) size += i;
  dst_val->clear();
  dst_val->resize(size, 0);

  size_t matched = 0;
  size_t grainsize = std::max(dst_key.size() / nthreads + 5,
                              static_cast<size_t>(1024*1024));
  KVMatchVaryLen<K, I, V>(src_key.begin(),
                          src_key.end(),
                          src_len.begin(),
                          src_val.begin(),
                          dst_key.begin(),
                          dst_key.end(),
                          dst_len->begin(),
                          dst_val->begin(),
                          op, grainsize, &matched);
  CHECK_EQ(matched, size);
  return size;
}

}  // namespace difacto

#endif  // DIFACTO_COMMON_KV_MATCH_H_
