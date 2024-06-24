#pragma once

#include "wpi/datalog/DataLog.h"

namespace wpi::log {
/**
 * Log arbitrary byte data.
 */
class RawLogEntry : public DataLogValueEntryImpl<std::vector<uint8_t>> {
 public:
  static constexpr std::string_view kDataType = "raw";

  RawLogEntry() = default;
  RawLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : RawLogEntry{log, name, {}, kDataType, timestamp} {}
  RawLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
              int64_t timestamp = 0)
      : RawLogEntry{log, name, metadata, kDataType, timestamp} {}
  RawLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
              std::string_view type, int64_t timestamp = 0)
      : DataLogValueEntryImpl{log, name, type, metadata, timestamp} {}

  /**
   * Appends a record to the log.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const uint8_t> data, int64_t timestamp = 0) {
    m_log->AppendRaw(m_entry, data, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const uint8_t> data, int64_t timestamp = 0);
};
}
