#pragma once
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <robot_common/hardware_interface/HybridJointInterface.h>
#include <realtime_tools/realtime_buffer.h>
#include <tron2_controllers/NANDU1TCPCommand.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <diagnostic_msgs/DiagnosticArray.h>
#include "NANDU1AdaptiveRuntime.h"

namespace tron2_controller {
class NANDU1AdaptiveController: public controller_interface::MultiInterfaceController<robot_common::HybridJointInterface,hardware_interface::ImuSensorInterface> {
  struct Command{std::array<double,3> value{};double received=0;bool invalid=false;};
  struct Health{std::array<double,3> age{1e9,1e9,1e9};double received=0;uint64_t epoch=0;};
  realtime_tools::RealtimeBuffer<Command> command_;
  realtime_tools::RealtimeBuffer<Health> health_;
  realtime_tools::RealtimeBuffer<int> requested_;
  std::vector<robot_common::HybridJointHandle> joints_;
  hardware_interface::ImuSensorHandle base_,up_;
  std::unique_ptr<nandu1::adaptive::Networks> networks_;
  nandu1::adaptive::Runtime runtime_;
  nandu1::Runtime sensors_;
  std::array<double,13> qTarget_{},dqTarget_{},lower_{},upper_{};
  std::array<double,3> applied_{};
  ros::Subscriber commandSub_,healthSub_,modeSub_;
  ros::Publisher trackingPub_,adaptivePub_,diagnosticsPub_,resetPub_;
  bool active_=false,initialized_=false,fault_=false,autoPolicy_=true,evidence_=true;
  double initialHeight_=0,nextPolicy_=0,lastPolicy_=0;
  int evaluationStage_=4;
  uint64_t epoch_=0,generation_=0;int64_t step_=0;
  std::string reason_;
  void reset();void damping();void fault(const std::string&);void diagnostic();
public:
  bool init(hardware_interface::RobotHW*,ros::NodeHandle&) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;
  void update(const ros::Time&,const ros::Duration&) override;
};
}
