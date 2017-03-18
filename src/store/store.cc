/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/store.h"
#include "./store_local.h"
#include "./kvstore_dist.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(KVStoreParam);

Store* Store::Create() {
  if (IsDistributed()) {
    return new KVStoreDist();
  } else {
    return new StoreLocal();
  }
}

}  // namespace difacto
