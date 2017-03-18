/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_NODE_ID_H_
#define DIFACTO_NODE_ID_H_
namespace difacto {

class NodeID {
 public:
  /** \brief node ID for the scheduler */
  static const int kScheduler = 1;
  /**
   * \brief the server node group ID
   *
   * group id can be combined:
   * - kServerGroup + kScheduler means all server nodes and the scheuduler
   * - kServerGroup + kWorkerGroup means all server and worker nodes
   */
  static const int kServerGroup = 2;
  /** \brief the worker node group ID */
  static const int kWorkerGroup = 4;

  static int Encode(int group, int rank) {
    return group + (rank+1) * 8;
  }

  /* \brief return the node group id */
  static int GetGroup(int id) {
    return (id % 8);
  }
};
}  // namespace difacto

#endif  // DIFACTO_NODE_ID_H_
