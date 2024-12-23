// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

// THIS FILE WAS AUTO-GENERATED BY ./wpiunits/generate_units.py. DO NOT MODIFY

package edu.wpi.first.units.measure;

import static edu.wpi.first.units.Units.*;
import edu.wpi.first.units.*;

@SuppressWarnings({"unchecked", "cast", "checkstyle", "PMD"})
public interface Current extends Measure<CurrentUnit> {
  static  Current ofRelativeUnits(double magnitude, CurrentUnit unit) {
    return new ImmutableCurrent(magnitude, unit.toBaseUnits(magnitude), unit);
  }

  static  Current ofBaseUnits(double baseUnitMagnitude, CurrentUnit unit) {
    return new ImmutableCurrent(unit.fromBaseUnits(baseUnitMagnitude), baseUnitMagnitude, unit);
  }

  @Override
  Current copy();

  @Override
  default MutCurrent mutableCopy() {
    return new MutCurrent(magnitude(), baseUnitMagnitude(), unit());
  }

  @Override
  CurrentUnit unit();

  @Override
  default CurrentUnit baseUnit() { return (CurrentUnit) unit().getBaseUnit(); }

  @Override
  default double in(CurrentUnit unit) {
    return unit.fromBaseUnits(baseUnitMagnitude());
  }

  @Override
  default Current unaryMinus() {
    return (Current) unit().ofBaseUnits(0 - baseUnitMagnitude());
  }

  @Override
  @Deprecated(since = "2025", forRemoval = true)
  @SuppressWarnings({"deprecation", "removal"})
  default Current negate() {
    return (Current) unaryMinus();
  }

  @Override
  default Current plus(Measure<? extends CurrentUnit> other) {
    return (Current) unit().ofBaseUnits(baseUnitMagnitude() + other.baseUnitMagnitude());
  }

  @Override
  default Current minus(Measure<? extends CurrentUnit> other) {
    return (Current) unit().ofBaseUnits(baseUnitMagnitude() - other.baseUnitMagnitude());
  }

  @Override
  default Current times(double multiplier) {
    return (Current) unit().ofBaseUnits(baseUnitMagnitude() * multiplier);
  }

  @Override
  default Current div(double divisor) {
    return (Current) unit().ofBaseUnits(baseUnitMagnitude() / divisor);
  }

  @Override
  default Velocity<CurrentUnit> per(TimeUnit period) {
    return div(period.of(1));
  }


  @Override
  default Mult<CurrentUnit, AccelerationUnit<?>> times(Acceleration<?> multiplier) {
    return (Mult<CurrentUnit, AccelerationUnit<?>>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, AccelerationUnit<?>> div(Acceleration<?> divisor) {
    return (Per<CurrentUnit, AccelerationUnit<?>>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, AngleUnit> times(Angle multiplier) {
    return (Mult<CurrentUnit, AngleUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, AngleUnit> div(Angle divisor) {
    return (Per<CurrentUnit, AngleUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, AngularAccelerationUnit> times(AngularAcceleration multiplier) {
    return (Mult<CurrentUnit, AngularAccelerationUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, AngularAccelerationUnit> div(AngularAcceleration divisor) {
    return (Per<CurrentUnit, AngularAccelerationUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, AngularMomentumUnit> times(AngularMomentum multiplier) {
    return (Mult<CurrentUnit, AngularMomentumUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, AngularMomentumUnit> div(AngularMomentum divisor) {
    return (Per<CurrentUnit, AngularMomentumUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, AngularVelocityUnit> times(AngularVelocity multiplier) {
    return (Mult<CurrentUnit, AngularVelocityUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, AngularVelocityUnit> div(AngularVelocity divisor) {
    return (Per<CurrentUnit, AngularVelocityUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, CurrentUnit> times(Current multiplier) {
    return (Mult<CurrentUnit, CurrentUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Dimensionless div(Current divisor) {
    return Value.of(baseUnitMagnitude() / divisor.baseUnitMagnitude());
  }

  @Override
  default Current div(Dimensionless divisor) {
    return (Current) Amps.of(baseUnitMagnitude() / divisor.baseUnitMagnitude());
  }

  @Override
  default Current times(Dimensionless multiplier) {
    return (Current) Amps.of(baseUnitMagnitude() * multiplier.baseUnitMagnitude());
  }


  @Override
  default Mult<CurrentUnit, DistanceUnit> times(Distance multiplier) {
    return (Mult<CurrentUnit, DistanceUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, DistanceUnit> div(Distance divisor) {
    return (Per<CurrentUnit, DistanceUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, EnergyUnit> times(Energy multiplier) {
    return (Mult<CurrentUnit, EnergyUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, EnergyUnit> div(Energy divisor) {
    return (Per<CurrentUnit, EnergyUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, ForceUnit> times(Force multiplier) {
    return (Mult<CurrentUnit, ForceUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, ForceUnit> div(Force divisor) {
    return (Per<CurrentUnit, ForceUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, FrequencyUnit> times(Frequency multiplier) {
    return (Mult<CurrentUnit, FrequencyUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, FrequencyUnit> div(Frequency divisor) {
    return (Per<CurrentUnit, FrequencyUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, LinearAccelerationUnit> times(LinearAcceleration multiplier) {
    return (Mult<CurrentUnit, LinearAccelerationUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, LinearAccelerationUnit> div(LinearAcceleration divisor) {
    return (Per<CurrentUnit, LinearAccelerationUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, LinearMomentumUnit> times(LinearMomentum multiplier) {
    return (Mult<CurrentUnit, LinearMomentumUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, LinearMomentumUnit> div(LinearMomentum divisor) {
    return (Per<CurrentUnit, LinearMomentumUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, LinearVelocityUnit> times(LinearVelocity multiplier) {
    return (Mult<CurrentUnit, LinearVelocityUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, LinearVelocityUnit> div(LinearVelocity divisor) {
    return (Per<CurrentUnit, LinearVelocityUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, MassUnit> times(Mass multiplier) {
    return (Mult<CurrentUnit, MassUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, MassUnit> div(Mass divisor) {
    return (Per<CurrentUnit, MassUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, MomentOfInertiaUnit> times(MomentOfInertia multiplier) {
    return (Mult<CurrentUnit, MomentOfInertiaUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, MomentOfInertiaUnit> div(MomentOfInertia divisor) {
    return (Per<CurrentUnit, MomentOfInertiaUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, MultUnit<?, ?>> times(Mult<?, ?> multiplier) {
    return (Mult<CurrentUnit, MultUnit<?, ?>>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, MultUnit<?, ?>> div(Mult<?, ?> divisor) {
    return (Per<CurrentUnit, MultUnit<?, ?>>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, PerUnit<?, ?>> times(Per<?, ?> multiplier) {
    return (Mult<CurrentUnit, PerUnit<?, ?>>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, PerUnit<?, ?>> div(Per<?, ?> divisor) {
    return (Per<CurrentUnit, PerUnit<?, ?>>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, PowerUnit> times(Power multiplier) {
    return (Mult<CurrentUnit, PowerUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, PowerUnit> div(Power divisor) {
    return (Per<CurrentUnit, PowerUnit>) Measure.super.div(divisor);
  }


  @Override
  default Voltage times(Resistance multiplier) {
    return Volts.of(baseUnitMagnitude() * multiplier.baseUnitMagnitude());
  }

  @Override
  default Per<CurrentUnit, ResistanceUnit> div(Resistance divisor) {
    return (Per<CurrentUnit, ResistanceUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, TemperatureUnit> times(Temperature multiplier) {
    return (Mult<CurrentUnit, TemperatureUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, TemperatureUnit> div(Temperature divisor) {
    return (Per<CurrentUnit, TemperatureUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, TimeUnit> times(Time multiplier) {
    return (Mult<CurrentUnit, TimeUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Velocity<CurrentUnit> div(Time divisor) {
    return VelocityUnit.combine(unit(), divisor.unit()).ofBaseUnits(baseUnitMagnitude() / divisor.baseUnitMagnitude());
  }


  @Override
  default Mult<CurrentUnit, TorqueUnit> times(Torque multiplier) {
    return (Mult<CurrentUnit, TorqueUnit>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, TorqueUnit> div(Torque divisor) {
    return (Per<CurrentUnit, TorqueUnit>) Measure.super.div(divisor);
  }


  @Override
  default Mult<CurrentUnit, VelocityUnit<?>> times(Velocity<?> multiplier) {
    return (Mult<CurrentUnit, VelocityUnit<?>>) Measure.super.times(multiplier);
  }

  @Override
  default Per<CurrentUnit, VelocityUnit<?>> div(Velocity<?> divisor) {
    return (Per<CurrentUnit, VelocityUnit<?>>) Measure.super.div(divisor);
  }


  @Override
  default Power times(Voltage multiplier) {
    return Watts.of(baseUnitMagnitude() * multiplier.baseUnitMagnitude());
  }

  @Override
  default Per<CurrentUnit, VoltageUnit> div(Voltage divisor) {
    return (Per<CurrentUnit, VoltageUnit>) Measure.super.div(divisor);
  }

}
