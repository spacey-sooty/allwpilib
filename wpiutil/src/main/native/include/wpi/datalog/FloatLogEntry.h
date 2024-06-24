#pragma once

#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log float values.
 */
class FloatLogEntry : public DataLogValueEntryImpl<float> {
 public:
  static constexpr std::string_view kDataType = "float";

  FloatLogEntry() = default;
  FloatLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : FloatLogEntry{log, name, {}, timestamp} {}
  FloatLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(float value, int64_t timestamp = 0) {
    m_log->AppendFloat(m_entry, value, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(float value, int64_t timestamp = 0) {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue != value) {
      m_lastValue = value;
      Append(value, timestamp);
    }
  }
};

/**
 * Log float values.
 */
class FloatLogEntry : public DataLogValueEntryImpl<float> {
 public:
  static constexpr std::string_view kDataType = "float";

  FloatLogEntry() = default;
  FloatLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : FloatLogEntry{log, name, {}, timestamp} {}
  FloatLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, kDataType, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(float value, int64_t timestamp = 0) {
    m_log->AppendFloat(m_entry, value, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param value Value to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(float value, int64_t timestamp = 0) {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue != value) {
      m_lastValue = value;
      Append(value, timestamp);
    }
  }
};
}
