package edu.wpi.first.apriltag.proto;

import edu.wpi.first.apriltag.AprilTagFields;
import edu.wpi.first.apriltag.proto.AprilTags.ProtobufAprilTagFields;
import edu.wpi.first.util.protobuf.Protobuf;

public class AprilTagFieldsProto implements Protobuf<AprilTagFields, ProtobufAprilTagFields> {
  @Override
  public Class<AprilTagFields> getTypeClass() {
    return AprilTagFields.class;
  }

  @Override
  public ProtobufAprilTagFields createMessage() {
    return ProtobufAprilTagFields.kDefaultField;
  }

  @Override
  public AprilTagFields unpack(ProtobufAprilTagFields msg) {
    switch (msg.getNumber()) {
      case 0:
        return AprilTagFields.k2022RapidReact;
        break;
      case 1:
        return AprilTagFields.k2023ChargedUp;
        break;
      case 2:
        return AprilTagFields.k2024Crescendo;
        break;
      default:
        return AprilTagFields.kDefaultField;
        break;
    }
  }

  @Override
  public void pack(ProtobufAprilTagFields msg, AprilTagFields value) {
  }
}
