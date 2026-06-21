// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "wpi/util/affinity.hpp"

#include <sched.h>
#include <unistd.h>

bool wpi::util::set_core_affinity(int core_id) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(core_id, &cpu_set);

  int rc = sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
  return static_cast<bool>(rc);
}
