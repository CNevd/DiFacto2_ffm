/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/reporter.h"
#include "./local_reporter.h"
#include "./dist_reporter.h"
namespace difacto {

Reporter* Reporter::Create() {
  if (IsDistributed()) {
    return new DistReporter();
  } else {
    return new LocalReporter();
  }
}

}  // namespace difacto
