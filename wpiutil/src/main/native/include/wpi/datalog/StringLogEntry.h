#pragma once

#include <string>
#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log string values.
 */
class StringLogEntry : public DataLogValueEntryImpl<std::string> {
 public:
  static constexpr const char* kDataType = "string";

  StringLogEntry() = default;
  StringLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : StringLogEntry{log, name, {}, kDataType, timestamp} {}
  StringLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                 int64_t timestamp = 0)
      : StringLogEntry{log, name, metadata, kDataType, timestamp} {}
  StringLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                 std::string_view type, int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, type, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::string_view value, int64_t timestamp = 0) {
    m_log->AppendString(m_entry, value, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::string value, int64_t timestamp = 0) {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue != value) {
      m_lastValue = value;
      Append(value, timestamp);
    }
  }
};
}
