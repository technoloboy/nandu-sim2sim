#include "tron2_controllers/NANDU1AdaptiveController.h"
#include <pluginlib/class_list_macros.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include "tron2_controllers/NANDU1AdaptiveProfiles.h"

namespace tron2_controller {
bool NANDU1AdaptiveController::init(hardware_interface::RobotHW* hw,ros::NodeHandle& nh){
  try{
    std::string task,validated,schema,ip,bundle;nh.getParam("task",task);nh.getParam("robot_ip",ip);nh.getParam("model_bundle",bundle);
    const auto profile=nandu1::adaptive::profileFor(task);
    if(ip!="127.0.0.1")throw std::runtime_error("Adaptive task and localhost required");
    nh.getParam("/model_bundle/task/task_id",validated);nh.getParam("/model_bundle/task/checkpoint_schema",schema);
    if(validated!=profile.canonical||schema!=profile.schema)throw std::runtime_error("Validated adaptive identity mismatch");
    int stage=-1;nh.getParam("/model_bundle/task/evaluation_stage",stage);
    if(stage!=profile.stage||!nh.getParam("/model_bundle/task/initial_up3_height",initialHeight_)||!std::isfinite(initialHeight_))throw std::runtime_error("Adaptive stage/height missing");
    evaluationStage_=stage;
    nh.param("auto_policy",autoPolicy_,true);
    auto* ji=hw->get<robot_common::HybridJointInterface>();auto* ii=hw->get<hardware_interface::ImuSensorInterface>();
    if(!ji||!ii)throw std::runtime_error("Missing adaptive hardware interfaces");
    for(const auto& name:nandu1::joints())joints_.push_back(ji->getHandle(name));
    base_=ii->getHandle("base_imu");up_=ii->getHandle("up3_imu");
    std::vector<double> lo,hi;nh.getParam("/model_bundle/task/joint_position_lower",lo);nh.getParam("/model_bundle/task/joint_position_upper",hi);
    if(lo.size()!=13||hi.size()!=13)throw std::runtime_error("Adaptive physical limits missing");
    for(int i=0;i<13;++i){if(!std::isfinite(lo[i])||!std::isfinite(hi[i])||lo[i]>=hi[i])throw std::runtime_error("Bad physical limits");lower_[i]=lo[i];upper_[i]=hi[i];}
    networks_=std::make_unique<nandu1::adaptive::Networks>(bundle,profile.width);
    runtime_=nandu1::adaptive::Runtime(profile.width,profile.heightMemory,profile.fasterHeight);
    command_.writeFromNonRT(Command{});health_.writeFromNonRT(Health{});requested_.writeFromNonRT(-1);
    commandSub_=nh.subscribe<tron2_controllers::NANDU1TCPCommand>(profile.topic,1,[this](const tron2_controllers::NANDU1TCPCommand::ConstPtr& m){
      Command c;c.value={m->relative_heading_rate,m->tcp_vx_base_body,m->tcp_height_world};c.received=ros::WallTime::now().toSec();
      c.invalid=!std::all_of(c.value.begin(),c.value.end(),[](double x){return std::isfinite(x);});command_.writeFromNonRT(c);
    });
    healthSub_=nh.subscribe<std_msgs::Float64MultiArray>("/nandu1/sensor_health",1,[this](const std_msgs::Float64MultiArray::ConstPtr& m){
      if(m->data.size()!=4||!std::all_of(m->data.begin(),m->data.end(),[](double x){return std::isfinite(x)&&x>=0;}))return;
      Health h;std::copy_n(m->data.begin(),3,h.age.begin());h.epoch=static_cast<uint64_t>(m->data[3]);h.received=ros::WallTime::now().toSec();health_.writeFromNonRT(h);
    });
    modeSub_=nh.subscribe<std_msgs::String>("/nandu1/set_mode",1,[this](const std_msgs::String::ConstPtr& m){
      if(m->data=="POLICY")requested_.writeFromNonRT(1);else if(m->data=="RESET")requested_.writeFromNonRT(2);else if(m->data=="PASSIVE")requested_.writeFromNonRT(0);
    });
    trackingPub_=nh.advertise<std_msgs::Float64MultiArray>("/nandu1/tracking",1);
    adaptivePub_=nh.advertise<std_msgs::Float64MultiArray>("/nandu1/adaptive/tracking",1);
    resetPub_=nh.advertise<std_msgs::Float64MultiArray>("/nandu1/reset_evidence",1,true);
    diagnosticsPub_=nh.advertise<diagnostic_msgs::DiagnosticArray>("/nandu1/diagnostics",1,true);
    reset();return true;
  }catch(const std::exception& e){ROS_ERROR("NANDU1 adaptive init: %s",e.what());return false;}
}
void NANDU1AdaptiveController::reset(){
  ++generation_;evidence_=true;runtime_.reset(initialHeight_,evaluationStage_);sensors_.reset();
  qTarget_=nandu1::defaults();dqTarget_.fill(0);applied_={0,0,initialHeight_};
  active_=initialized_=fault_=false;reason_.clear();step_=0;nextPolicy_=lastPolicy_=0;command_.writeFromNonRT(Command{});
}
void NANDU1AdaptiveController::starting(const ros::Time&){reset();}
void NANDU1AdaptiveController::stopping(const ros::Time&){active_=false;damping();diagnostic();}
void NANDU1AdaptiveController::damping(){for(size_t i=0;i<joints_.size();++i)joints_[i].setCommand(joints_[i].getPosition(),0,0,nandu1::kd()[i],0,0);}
void NANDU1AdaptiveController::fault(const std::string& reason){if(!fault_)ROS_ERROR("NANDU1 adaptive FAULT: %s",reason.c_str());fault_=true;active_=false;reason_=reason;}
void NANDU1AdaptiveController::diagnostic(){
  diagnostic_msgs::DiagnosticArray d;d.header.stamp=ros::Time::now();diagnostic_msgs::DiagnosticStatus s;
  s.name="nandu1/policy";s.level=fault_?2:0;s.message=fault_?reason_:(active_?"POLICY":"PASSIVE");
  for(const auto& entry:std::vector<std::pair<std::string,std::string>>{
      {"mode",fault_?"FAULT":(active_?"POLICY":"PASSIVE")},
      {"epoch",std::to_string(epoch_)},{"generation",std::to_string(generation_)}}){
    diagnostic_msgs::KeyValue v;v.key=entry.first;v.value=entry.second;s.values.push_back(v);
  }
  d.status.push_back(s);diagnosticsPub_.publish(d);
}
void NANDU1AdaptiveController::update(const ros::Time&,const ros::Duration&){
  const double now=ros::WallTime::now().toSec();const Health h=*health_.readFromRT();int requested=*requested_.readFromRT();
  if(requested==2){reset();requested=1;requested_.writeFromNonRT(1);}
  if(h.epoch>0&&h.epoch!=epoch_){epoch_=h.epoch;reset();}
  if(requested==0){active_=false;initialized_=false;damping();diagnostic();return;}
  bool healthy=h.epoch>0&&now-h.received<.1;for(double age:h.age)healthy=healthy&&age+now-h.received<.1;
  if(!healthy){if(initialized_)fault("Sensor state expired");damping();diagnostic();return;}
  initialized_=true;
  if(requested==-1&&!autoPolicy_){active_=false;damping();diagnostic();return;}
  if(fault_){damping();diagnostic();return;}
  // Every entry is a new command session, including a short PASSIVE roundtrip.
  // Do not reuse a still-fresh external command or the previous mouse reset key.
  if(!active_){reset();initialized_=true;active_=true;nextPolicy_=now;ROS_INFO("NANDU1 E/r2 sensors ready -> POLICY");}
  try{
    std::array<double,13> q,dq;
    for(int i=0;i<13;++i){q[i]=joints_[i].getPosition();dq[i]=joints_[i].getVelocity();
      if(!std::isfinite(q[i])||!std::isfinite(dq[i])||q[i]<lower_[i]-.05||q[i]>upper_[i]+.05)throw std::runtime_error("Joint state outside physical limits");}
    if(now>=nextPolicy_){
      nextPolicy_+=.02;if(now-nextPolicy_>=.02)nextPolicy_=now+.02;
      auto quat=[](const hardware_interface::ImuSensorHandle& i){const auto* p=i.getOrientation();return Eigen::Quaterniond(p[3],p[0],p[1],p[2]);};
      auto gyro=[](const hardware_interface::ImuSensorHandle& i){const auto* p=i.getAngularVelocity();return Eigen::Vector3d(p[0],p[1],p[2]);};
      auto obs=sensors_.observe(quat(base_),gyro(base_),quat(up_),gyro(up_),q,dq);
      const auto command=*command_.readFromRT();if(command.invalid)throw std::runtime_error("Nonfinite external command");
      bool expired=command.received==0||now-command.received>.5;
      auto requestedIntent=command.value;if(expired)requestedIntent={0,0,applied_[2]};
      nandu1::adaptive::Vec intent;for(double x:requestedIntent)intent.push_back(x);
      auto result=runtime_.step(obs,intent,step_,*networks_);
      auto targets=sensors_.decode(result.action);size_t projected=nandu1::clipPositionTargets(targets.first,lower_,upper_);
      runtime_.reportIntervalClipping(false,projected>0);qTarget_=targets.first;dqTarget_=targets.second;sensors_.applied(result.action);
      std::copy_n(result.commands.begin(),3,applied_.begin());
      if(evidence_){
        double lastInf=0,repeat=0;for(int i=36;i<49;++i)lastInf=std::max(lastInf,std::abs(double(obs[i])));
        for(int i=0;i<245;++i)repeat=std::max(repeat,std::abs(double(result.history[i]-result.history[i%49])));
        std_msgs::Float64MultiArray e;e.data={now,double(epoch_),double(generation_),lastInf,repeat,0,0,0,0,0,initialHeight_,initialHeight_};resetPub_.publish(e);evidence_=false;
      }
      std_msgs::Float64MultiArray legacy;legacy.data={now,double(epoch_)};
      legacy.data.insert(legacy.data.end(),applied_.begin(),applied_.end());legacy.data.insert(legacy.data.end(),result.encoded.begin(),result.encoded.end());
      legacy.data.insert(legacy.data.end(),result.action.begin(),result.action.end());
      for(const auto& values:{q,dq,qTarget_,dqTarget_})legacy.data.insert(legacy.data.end(),values.begin(),values.end());
      for(const auto& joint:joints_)legacy.data.push_back(joint.getEffort());
      for(int i=0;i<13;++i){double raw=nandu1::kp()[i]*(qTarget_[i]-q[i])+nandu1::kd()[i]*(dqTarget_[i]-dq[i]),limited=std::clamp(raw,-nandu1::effort()[i],nandu1::effort()[i]);legacy.data.push_back(limited);legacy.data.push_back(raw!=limited);}
      legacy.data.push_back(result.heading);legacy.data.push_back(result.headingTarget);
      legacy.data.insert(legacy.data.end(),requestedIntent.begin(),requestedIntent.end());trackingPub_.publish(legacy);
      std_msgs::Float64MultiArray full=legacy;
      full.data.insert(full.data.end(),{double(step_),lastPolicy_?now-lastPolicy_:.02,double(expired),double(projected),nandu1::adaptive::project(obs).tilt});
      full.data.insert(full.data.end(),requestedIntent.begin(),requestedIntent.end());
      for(const auto& values:{result.dynamic,result.hidden,result.eta,result.actorInput})full.data.insert(full.data.end(),values.begin(),values.end());
      adaptivePub_.publish(full);diagnostic();lastPolicy_=now;++step_;
    }
    for(int i=0;i<13;++i)joints_[i].setCommand(qTarget_[i],dqTarget_[i],nandu1::kp()[i],nandu1::kd()[i],0,0);
  }catch(const std::exception& e){fault(e.what());damping();diagnostic();}
}
}
PLUGINLIB_EXPORT_CLASS(tron2_controller::NANDU1AdaptiveController,controller_interface::ControllerBase)
