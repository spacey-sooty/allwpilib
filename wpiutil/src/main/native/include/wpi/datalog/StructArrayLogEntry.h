#pragma once

#include "wpi/datalog/DataLog.h"
#include "wpi/struct/Struct.h"
namespace wpi::log {
/**
 * Log raw struct serializable array of objects.
 */
template <typename T, typename... I>
  requires StructSerializable<T, I...>
class StructArrayLogEntry : public DataLogEntry {
  using S = Struct<T, I...>;

 public:
  StructArrayLogEntry() = default;
  StructArrayLogEntry(DataLog& log, std::string_view name, I... info,
                      int64_t timestamp = 0)
      : StructArrayLogEntry{log, name, {}, std::move(info)..., timestamp} {}
  StructArrayLogEntry(DataLog& log, std::string_view name,
                      std::string_view metadata, I... info,
                      int64_t timestamp = 0)
      : m_info{std::move(info)...} {
    m_log = &log;
    log.AddStructSchema<T, I...>(info..., timestamp);
    m_entry = log.Start(
        name, MakeStructArrayTypeString<T, std::dynamic_extent>(info...),
        metadata, timestamp);
  }

  StructArrayLogEntry(StructArrayLogEntry&& rhs)
      : DataLogEntry{std::move(rhs)},
        m_buf{std::move(rhs.m_buf)},
        m_info{std::move(rhs.m_info)} {
    std::scoped_lock lock{rhs.m_mutex};
    m_lastValue = std::move(rhs.m_lastValue);
  }
  StructArrayLogEntry& operator=(StructArrayLogEntry&& rhs) {
    DataLogEntry::operator=(std::move(rhs));
    m_buf = std::move(rhs.m_buf);
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
  template <typename U>
#if __cpp_lib_ranges >= 201911L
    requires std::ranges::range<U> &&
             std::convertible_to<std::ranges::range_value_t<U>, T>
#endif
  void Append(U&& data, int64_t timestamp = 0) {
    std::apply(
        [&](const I&... info) {
          m_buf.Write(
              std::forward<U>(data),
              [&](auto bytes) { m_log->AppendRaw(m_entry, bytes, timestamp); },
              info...);
        },
        m_info);
  }

  /**
   * Appends a record to the log.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Append(std::span<const T> data, int64_t timestamp = 0) {
    std::apply(
        [&](const I&... info) {
          m_buf.Write(
              data,
              [&](auto bytes) { m_log->AppendRaw(m_entry, bytes, timestamp); },
              info...);
        },
        m_info);
  }

  /**
   * Updates the last value and appends a record to the log if it has changed.
   *
   * @param data Data to record
   * @param timestamp Time stamp (may be 0 to indicate now)
   */
  void Update(std::span<const T> data, int64_t timestamp = 0) {
    std::apply(
        [&](const I&... info) {
          m_buf.Write(
              data,
              [&](auto bytes) {
                std::scoped_lock lock{m_mutex};
                if (m_lastValue.empty() ||
                    !std::equal(bytes.begin(), bytes.end(), m_lastValue.begin(),
                                m_lastValue.end())) {
                  m_lastValue.assign(bytes.begin(), bytes.end());
                  m_log->AppendRaw(m_entry, bytes, timestamp);
                }
              },
              info...);
        },
        m_info);
  }

  /**
   * Gets the last value.  Note that Append() calls do not update the last
   * value.
   *
   * @return Last value (empty if no last value)
   */
  std::optional<std::vector<T>> GetLastValue() const {
    std::scoped_lock lock{m_mutex};
    if (m_lastValue.empty()) {
      return std::nullopt;
    }
    size_t size = std::apply(S::GetSize, m_info);
    std::vector<T> rv;
    rv.value.reserve(m_lastValue.size() / size);
    for (auto in = m_lastValue.begin(), end = m_lastValue.end(); in < end;
         in += size) {
      std::apply(
          [&](const I&... info) {
            rv.value.emplace_back(S::Unpack(
                std::span<const uint8_t>{std::to_address(in), size}, info...));
          },
          m_info);
    }
    return rv;
  }

 private:
  mutable wpi::mutex m_mutex;
  StructArrayBuffer<T, I...> m_buf;
  std::vector<uint8_t> m_lastValue;
  [[no_unique_address]]
  std::tuple<I...> m_info;
};
}
