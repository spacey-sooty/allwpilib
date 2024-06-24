#pragma once

#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log double values.
 */
class DoubleLogEntry : public DataLogValueEntryImpl<double> {
 public:
  static constexpr std::string_view kDataType = "double";

  DoubleLogEntry() = default;
  DoubleLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : DoubleLogEntry{log, name, {}, timestamp} {}
  DoubleLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                 int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(double value, int64_t timestamp = 0) {
    m_log->AppendDouble(m_entry, value, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(double value, int64_t timestamp = 0) {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue != value) {
      m_lastValue = value;
      Append(value, timestamp);
    }
  }
};
}
