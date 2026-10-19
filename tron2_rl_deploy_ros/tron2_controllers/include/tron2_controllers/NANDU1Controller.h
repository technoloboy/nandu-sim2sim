#pragma once
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <robot_common/hardware_interface/HybridJointInterface.h>
#include <realtime_tools/realtime_buffer.h>
#include <tron2_controllers/NANDU1Command.h>
#include <tron2_controllers/NANDU1V14Command.h>
#include <tron2_controllers/NANDU1TCPCommand.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <diagnostic_msgs/DiagnosticArray.h>
#include "NANDU1Onnx.h"
#include "NANDU1V14.h"
#include "NANDU1V141.h"
#include "NANDU1Profiles.h"

namespace tron2_controller {
class NANDU1Controller : public controller_interface::MultiInterfaceController<robot_common::HybridJointInterface,hardware_interface::ImuSensorInterface> {
  struct Command {std::array<double,3> v{};double received=0;bool invalid=false;};
  struct Health {std::array<double,3> age{1e9,1e9,1e9};double received=0;uint64_t epoch=0;};
  realtime_tools::RealtimeBuffer<Command> command_;
  realtime_tools::RealtimeBuffer<Health> health_;
  realtime_tools::RealtimeBuffer<int> requested_;
  std::vector<robot_common::HybridJointHandle> joints_;
  hardware_interface::ImuSensorHandle base_,up_;
  std::unique_ptr<nandu1::Network> encoder_,actor_;
  nandu1::Runtime runtime_;
  nandu1::v14::CommandProcessor processor_;
  nandu1::v141::CommandProcessor smoothProcessor_;
  nandu1::v131::CommandProcessor tcpProcessor_;
  nandu1::Profile profile_;
  bool v14_=false;
  std::array<double,3> commandLower_{-1.57,-1,.4},commandUpper_{1.57,1,1.4},commandRate_{2,.8,.5};
  std::array<double,3> externalApplied_{};
  std::array<double,13> qTarget_{},dqTarget_{},lower_{},upper_{};
  std::array<double,3> appliedCommand_{};
  std::vector<float> estimate_=std::vector<float>(7,0),actions_=std::vector<float>(13,0);
  ros::Subscriber commandSub_,healthSub_,modeSub_;
  ros::Publisher trackingPub_,diagnosticsPub_,resetEvidencePub_;
  double initialHeight_=0,nextPolicy_=0;
  uint64_t epoch_=0;
  uint64_t resetGeneration_=0;
  bool resetEvidencePending_=true;
  bool initialized_=false,active_=false,fault_=false,autoPolicy_=false;
  std::string faultReason_;
  void reset();
  void fault(const std::string& reason);
  void damping();
  void stateDiagnostic();
public:
  bool init(hardware_interface::RobotHW*,ros::NodeHandle&) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;
  void update(const ros::Time&,const ros::Duration&) override;
};
}
