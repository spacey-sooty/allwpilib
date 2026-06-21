// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "wpi/util/affinity.hpp"

#include <windows.h>

bool wpi::util::set_core_affinity(int core_id) {
  DWORD_PTR affinity_mask = (DWORD_PTR)1 << core_id;
  DWORD_PTR previous_mask =
      SetThreadAffinityMask(GetCurrentThread(), affinity_mask);

  return static_cast<bool>(previous_mask);
}
