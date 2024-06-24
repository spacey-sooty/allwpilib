#pragma once
#include <vector>
#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log array of float values.
 */
class FloatArrayLogEntry : public DataLogValueEntryImpl<std::vector<float>> {
 public:
  static constexpr const char* kDataType = "float[]";

  FloatArrayLogEntry() = default;
  FloatArrayLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : FloatArrayLogEntry{log, name, {}, timestamp} {}
  FloatArrayLogEntry(DataLog& log, std::string_view name,
                     std::string_view metadata, int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const float> arr, int64_t timestamp = 0) {
    m_log->AppendFloatArray(m_entry, arr, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::initializer_list<float> arr, int64_t timestamp = 0) {
    Append({arr.begin(), arr.end()}, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const float> arr, int64_t timestamp = 0);

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::initializer_list<float> arr, int64_t timestamp = 0) {
    Update({arr.begin(), arr.end()}, timestamp);
  }
};
}
