// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

namespace wpi::util {
/**
 * Pins the current thread to run on the provided core.
 * @param core_id The core ID to pin the current thread to.
 * @return True if setting the core affinity succeedds and false if it fails.
 */
bool set_core_affinity(int core_id);
}  // namespace wpi::util
