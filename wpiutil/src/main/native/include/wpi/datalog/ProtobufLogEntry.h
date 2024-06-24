#pragma once

#include "wpi/datalog/DataLog.h"
#include "wpi/protobuf/Protobuf.h"

namespace wpi::log {
/**
 * Log protobuf serializable objects.
 */
template <ProtobufSerializable T>
class ProtobufLogEntry : public DataLogEntry {
  using P = Protobuf<T>;

 public:
  ProtobufLogEntry() = default;
  ProtobufLogEntry(DataLog& log, std::string_view name, int64_t timestamp = 0)
      : ProtobufLogEntry{log, name, {}, timestamp} {}
  ProtobufLogEntry(DataLog& log, std::string_view name,
                   std::string_view metadata, int64_t timestamp = 0) {
    m_log = &log;
    log.AddProtobufSchema<T>(m_msg, timestamp);
    m_entry = log.Start(name, m_msg.GetTypeString(), metadata, timestamp);
  }

  /**
   * Appends a record to the log.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(const T& data, int64_t timestamp = 0) {
    SmallVector<uint8_t, 128> buf;
    {
      std::scoped_lock lock{m_mutex};
      m_msg.Pack(buf, data);
    }
    m_log->AppendRaw(m_entry, buf, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(const T& data, int64_t timestamp = 0) {
    std::scoped_lock lock{m_mutex};
    wpi::SmallVector<uint8_t, 128> buf;
    m_msg.Pack(buf, data);
    if (m_lastValue.empty() ||
        !std::equal(buf.begin(), buf.end(), m_lastValue.begin(),
                    m_lastValue.end())) {
      m_lastValue.assign(buf.begin(), buf.end());
      m_log->AppendRaw(m_entry, buf, timestamp);
    }
  }

  /**
   * Gets the last value.  Note that Append() calls do not update the last
   * value.
   *
   * @return Last value (empty if no last value)
   */
  std::optional<T> GetLastValue() const {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue.empty()) {
      return std::nullopt;
    }
    return m_msg.Unpack(m_lastValue);
  }

 private:
  mutable wpi::mutex m_mutex;
  ProtobufMessage<T> m_msg;
  std::vector<uint8_t> m_lastValue;
};

}
