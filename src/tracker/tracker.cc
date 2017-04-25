/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/tracker.h"
#include "./local_tracker.h"
#include "./dist_tracker.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(WorkloadPoolParam);

Tracker* Tracker::Create() {
  if (IsDistributed()) {
    return new DistTracker();
  } else {
    return new LocalTracker();
  }
}

}  // namespace difacto
