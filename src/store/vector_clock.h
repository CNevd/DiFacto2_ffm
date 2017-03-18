#ifndef DIFACTO_VECTOR_CLOCK_H_
#define DIFACTO_VECTOR_CLOCK_H_
#include <string>
#include <vector>
#include <limits>

namespace difacto {

class VectorClock {
 public:
  VectorClock(int n) :
    local_clock_(n, 0), global_clock_(0), size_(0) {}

  /*! brief update local and global clock */
  bool Update(int i) {
    ++local_clock_[i];
    if (global_clock_ < *(std::min_element(std::begin(local_clock_),
      std::end(local_clock_)))) {
      ++global_clock_;
      if (global_clock_ == max_element()) {
        return true;
      }
    }
    return false;
  }

  std::string DebugString() {
    std::string os = "global ";
    os += std::to_string(global_clock_) + " local: ";
    for (auto i : local_clock_) { 
      if (i == std::numeric_limits<int>::max()) os += "-1 ";
      else os += std::to_string(i) + " ";
    }
    return os;
  }
  /*! \brief get the local clock for a worker */
  int local_clock(int i) const { return local_clock_[i]; }
  /*! \brief get the global clock for this server */
  int global_clock() const { return global_clock_; }

 private:
  int max_element() const {
    int max = global_clock_;
    for (auto val : local_clock_) {
      max = (val != std::numeric_limits<int>::max() && val > max) ? val : max;
    }
    return max;
  }

  int min_element() const {
    return *std::min_element(std::begin(local_clock_), std::end(local_clock_));
  }

  std::vector<int> local_clock_;
  int global_clock_;
  int size_;

};

} // namespace difacto


#endif // DIFACTO_VECTOR_CLOCK_H_


