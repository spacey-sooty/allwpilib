// package edu.wpi.first.apriltag.proto;
//
// import java.util.ArrayList;
// import java.util.List;
//
// import edu.wpi.first.apriltag.AprilTag;
// import edu.wpi.first.apriltag.AprilTagFieldLayout;
// import edu.wpi.first.apriltag.proto.AprilTags.ProtobufAprilTag;
// import edu.wpi.first.apriltag.proto.AprilTags.ProtobufAprilTagFieldLayout;
// import edu.wpi.first.math.geometry.Pose3d;
// import edu.wpi.first.util.protobuf.Protobuf;
// import us.hebi.quickbuf.Descriptors.Descriptor;
//
// public class AprilTagFieldLayoutProto implements Protobuf<AprilTagFieldLayout,
// ProtobufAprilTagFieldLayout> {
//    @Override
//    public Class<AprilTagFieldLayout> getTypeClass() {
//        return AprilTagFieldLayout.class;
//    }
//
//    @Override
//    public Descriptor getDescriptor() {
//        return ProtobufAprilTagFieldLayout.getDescriptor();
//    }
//
//  @Override
//  public ProtobufAprilTagFieldLayout createMessage() {
//    return ProtobufAprilTagFieldLayout.newInstance();
//  }
//
//  @Override
//  public void pack(ProtobufAprilTagFieldLayout msg, AprilTagFieldLayout value) {
//    Pose3d.proto.pack(msg.getOrigin(), value.getOrigin());
//    //TODO!
//  }
//
//  @Override
//  public AprilTagFieldLayout unpack(ProtobufAprilTagFieldLayout msg) {
//      List<ProtobufAprilTag> tags = new ArrayList<>();
//      msg.getMutableMApriltags().forEach((tag) -> tags.add(tag.getValue()));
//      List<AprilTag> tags2 = new ArrayList<>();
//      tags.forEach((tag) -> tags2.add(new AprilTag(tag.getID(), pose));
//      return new AprilTagFieldLayout(, msg.getFieldDimensions().getLength(),
// msg.getFieldDimensions().getWidth());
//  }
// }
