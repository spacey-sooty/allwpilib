package edu.wpi.first.apriltag.proto;

import edu.wpi.first.apriltag.AprilTag;
import edu.wpi.first.apriltag.proto.AprilTags.ProtobufAprilTag;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.util.protobuf.Protobuf;
import us.hebi.quickbuf.Descriptors.Descriptor;

public class AprilTagProto implements Protobuf<AprilTag, ProtobufAprilTag> {
  @Override
  public Class<AprilTag> getTypeClass() {
    return AprilTag.class;
  }

  @Override
  public Descriptor getDescriptor() {
    return ProtobufAprilTag.getDescriptor();
  }

  @Override
  public Protobuf<?, ?>[] getNested() {
    return new Protobuf<?, ?>[] {Pose3d.proto};
  }

  @Override
  public ProtobufAprilTag createMessage() {
    return ProtobufAprilTag.newInstance();
  }

  @Override
  public AprilTag unpack(ProtobufAprilTag msg) {
    return new AprilTag(msg.getID(), Pose3d.proto.unpack(msg.getPose()));
  }

  @Override
  public void pack(ProtobufAprilTag msg, AprilTag value) {
    msg.setID(value.ID);
    Pose3d.proto.pack(msg.getPose(), value.pose);
  }
}
