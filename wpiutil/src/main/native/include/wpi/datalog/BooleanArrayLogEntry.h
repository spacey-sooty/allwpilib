#pragma once

#include <vector>
#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log array of boolean values.
 */
class BooleanArrayLogEntry : public DataLogValueEntryImpl<std::vector<int>> {
 public:
  static constexpr const char* kDataType = "boolean[]";

  BooleanArrayLogEntry() = default;
  BooleanArrayLogEntry(DataLog& log, std::string_view name,
                       int64_t timestamp = 0)
      : BooleanArrayLogEntry{log, name, {}, timestamp} {}
  BooleanArrayLogEntry(DataLog& log, std::string_view name,
                       std::string_view metadata, int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.  For find functions to work, timestamp
   * must be monotonically increasing.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const bool> arr, int64_t timestamp = 0) {
    m_log->AppendBooleanArray(m_entry, arr, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::initializer_list<bool> arr, int64_t timestamp = 0) {
    Append(std::span{arr.begin(), arr.end()}, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const int> arr, int64_t timestamp = 0) {
    m_log->AppendBooleanArray(m_entry, arr, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::initializer_list<int> arr, int64_t timestamp = 0) {
    Append(std::span{arr.begin(), arr.end()}, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const uint8_t> arr, int64_t timestamp = 0) {
    m_log->AppendBooleanArray(m_entry, arr, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const bool> arr, int64_t timestamp = 0);

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::initializer_list<bool> arr, int64_t timestamp = 0) {
    Update(std::span{arr.begin(), arr.end()}, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const int> arr, int64_t timestamp = 0);

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::initializer_list<int> arr, int64_t timestamp = 0) {
    Update(std::span{arr.begin(), arr.end()}, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const uint8_t> arr, int64_t timestamp = 0);
};
}
