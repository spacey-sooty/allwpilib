//package edu.wpi.first.apriltag.proto;
//
//import edu.wpi.first.apriltag.AprilTagFields;
//import edu.wpi.first.apriltag.proto.AprilTags.ProtobufAprilTagFields;
//import edu.wpi.first.util.protobuf.Protobuf;
//import us.hebi.quickbuf.Descriptors.Descriptor;
//
//public class AprilTagFieldsProto implements Protobuf<AprilTagFields, ProtobufAprilTagFields> {
//    @Override
//    public Class<AprilTagFields> getTypeClass() {
//        return ProtobufAprilTagFields.class;
//    }
//
//    @Override
//    public Descriptor getDescriptor() {
//        return ProtobufAprilTagFields.getDescriptor();
//    }
//
//  @Override
//  public ProtobufAprilTagFields createMessage() {
//    return ProtobufAprilTagFields.newInstance();
//  }
//
//  @Override
//  public void pack(ProtobufAprilTagFields msg, AprilTagFields value) {
//  }
//}
