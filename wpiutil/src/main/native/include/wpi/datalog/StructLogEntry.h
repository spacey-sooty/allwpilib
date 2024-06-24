#pragma once


#include "wpi/datalog/DataLog.h"
#include "wpi/struct/Struct.h"
namespace wpi::log {
/**
 * Log raw struct serializable objects.
 */
template <typename T, typename... I>
  requires StructSerializable<T, I...>
class StructLogEntry : public DataLogEntry {
  using S = Struct<T, I...>;

 public:
  StructLogEntry() = default;
  StructLogEntry(DataLog& log, std::string_view name, I... info,
                 int64_t timestamp = 0)
      : StructLogEntry{log, name, {}, std::move(info)..., timestamp} {}
  StructLogEntry(DataLog& log, std::string_view name, std::string_view metadata,
                 I... info, int64_t timestamp = 0)
      : m_info{std::move(info)...} {
    m_log = &log;
    log.AddStructSchema<T, I...>(info..., timestamp);
    m_entry = log.Start(name, S::GetTypeString(info...), metadata, timestamp);
  }

  StructLogEntry(StructLogEntry&& rhs)
      : DataLogEntry{std::move(rhs)}, m_info{std::move(rhs.m_info)} {
    std::scoped_lock lock{rhs.m_mutex};
    m_lastValue = std::move(rhs.m_lastValue);
  }
  StructLogEntry& operator=(StructLogEntry&& rhs) {
    DataLogEntry::operator=(std::move(rhs));
    m_info = std::move(rhs.m_info);
    std::scoped_lock lock{m_mutex, rhs.m_mutex};
    m_lastValue = std::move(rhs.m_lastValue);
    return *this;
  }

  /**
   * Appends a record to the log.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(const T& data, int64_t timestamp = 0) {
    if constexpr (sizeof...(I) == 0) {
      if constexpr (wpi::is_constexpr([] { S::GetSize(); })) {
        uint8_t buf[S::GetSize()];
        S::Pack(buf, data);
        m_log->AppendRaw(m_entry, buf, timestamp);
        return;
      }
    }
    wpi::SmallVector<uint8_t, 128> buf;
    buf.resize_for_overwrite(std::apply(S::GetSize, m_info));
    std::apply([&](const I&... info) { S::Pack(buf, data, info...); }, m_info);
    m_log->AppendRaw(m_entry, buf, timestamp);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(const T& data, int64_t timestamp = 0) {
    if constexpr (sizeof...(I) == 0) {
      if constexpr (wpi::is_constexpr([] { S::GetSize(); })) {
        uint8_t buf[S::GetSize()];
        S::Pack(buf, data);
        std::scoped_lock lock{m_mutex};
        if (m_lastValue.empty() ||
            !std::equal(buf, buf + S::GetSize(), m_lastValue.begin(),
                        m_lastValue.end())) {
          m_lastValue.assign(buf, buf + S::GetSize());
          m_log->AppendRaw(m_entry, buf, timestamp);
        }
        return;
      }
    }
    wpi::SmallVector<uint8_t, 128> buf;
    buf.resize_for_overwrite(std::apply(S::GetSize, m_info));
    std::apply([&](const I&... info) { S::Pack(buf, data, info...); }, m_info);
    std::scoped_lock lock{m_mutex};
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
    return std::apply(
        [&](const I&... info) { S::Unpack(m_lastValue, info...); }, m_info);
  }

 private:
  mutable wpi::mutex m_mutex;
  std::vector<uint8_t> m_lastValue;
  [[no_unique_address]]
  std::tuple<I...> m_info;
};

}
