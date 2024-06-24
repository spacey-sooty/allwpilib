#pragma once

#include <vector>
#include <string>
#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log array of string values.
 */
class StringArrayLogEntry
    : public DataLogValueEntryImpl<std::vector<std::string>> {
 public:
  static constexpr const char* kDataType = "string[]";

  StringArrayLogEntry() = default;
  StringArrayLogEntry(DataLog& log, std::string_view name,
                      int64_t timestamp = 0)
      : StringArrayLogEntry{log, name, {}, timestamp} {}
  StringArrayLogEntry(DataLog& log, std::string_view name,
                      std::string_view metadata, int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const std::string> arr, int64_t timestamp = 0) {
    m_log->AppendStringArray(m_entry, arr, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const std::string_view> arr, int64_t timestamp = 0) {
    m_log->AppendStringArray(m_entry, arr, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::initializer_list<std::string_view> arr,
              int64_t timestamp = 0) {
    Append(std::span<const std::string_view>{arr.begin(), arr.end()},
           timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const std::string> arr, int64_t timestamp = 0);

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const std::string_view> arr, int64_t timestamp = 0);

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param arr Values to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::initializer_list<std::string_view> arr,
              int64_t timestamp = 0) {
    Update(std::span<const std::string_view>{arr.begin(), arr.end()},
           timestamp);
  }
};
}

