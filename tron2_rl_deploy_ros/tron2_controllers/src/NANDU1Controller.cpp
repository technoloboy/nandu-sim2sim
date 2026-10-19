#include "tron2_controllers/NANDU1Controller.h"
#include <pluginlib/class_list_macros.h>
#include <urdf/model.h>
#include <diagnostic_msgs/DiagnosticStatus.h>

namespace tron2_controller {
bool NANDU1Controller::init(hardware_interface::RobotHW* hw,ros::NodeHandle& nh){
  try{
    std::string bundle,ip,task;
    nh.param<std::string>("model_bundle",bundle,"");
    nh.param<std::string>("robot_ip",ip,"");
    nh.param<std::string>("task",task,"");
    profile_=nandu1::selectProfile(task);
    v14_=profile_.actorWidth==62;
    if(ip!="127.0.0.1")throw std::runtime_error("NANDU1 requires localhost");
    nh.param("auto_policy",autoPolicy_,false);
    if(!nh.getParam("/model_bundle/task/initial_up3_height",initialHeight_))throw std::runtime_error("Missing initial height");
    std::string validatedTask;
    if(!nh.getParam("/model_bundle/task/task_id",validatedTask) || validatedTask!=profile_.task)throw std::runtime_error("Validated bundle task mismatch");
    if(profile_.networkErrors){
      std::string schema;nh.getParam("/model_bundle/task/checkpoint_schema",schema);
      if(schema!="nandu_network_v1-3-1_2_E_r1")throw std::runtime_error("Original-E checkpoint schema required");
    }
    int stage=4;nh.getParam("/model_bundle/task/evaluation_stage",stage);
    if(stage<0||stage>4)throw std::runtime_error("Invalid evaluation stage");
    processor_.configure(stage,initialHeight_);smoothProcessor_.configure(stage,initialHeight_);tcpProcessor_.configure(stage,initialHeight_);
    auto parameter=[&](const char* name,std::array<double,3>& out,bool required){
      std::vector<double> value;
      if(!nh.getParam(std::string("/model_bundle/task/")+name,value)){if(required)throw std::runtime_error(std::string("Missing ")+name);return;}
      if(value.size()!=3||!std::all_of(value.begin(),value.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("Invalid command bounds");
      std::copy_n(value.begin(),3,out.begin());
    };
    parameter("command_min",commandLower_,true);parameter("command_max",commandUpper_,true);parameter("command_rate",commandRate_,false);
    for(int i=0;i<3;++i)if(commandLower_[i]>=commandUpper_[i]||commandRate_[i]<=0)throw std::runtime_error("Invalid command bounds/rates");
    auto* ji=hw->get<robot_common::HybridJointInterface>();
    auto* ii=hw->get<hardware_interface::ImuSensorInterface>();
    if(!ji || !ii)throw std::runtime_error("Missing hardware interface");
    for(const auto& name:nandu1::joints())joints_.push_back(ji->getHandle(name));
    base_=ii->getHandle("base_imu");up_=ii->getHandle("up3_imu");
    // Limits come from the authoritative USD-derived MJCF, not legacy URDF limits.
    std::vector<double> lo,hi;
    if(!nh.getParam("/model_bundle/task/joint_position_lower",lo) || !nh.getParam("/model_bundle/task/joint_position_upper",hi)
       || lo.size()!=13 || hi.size()!=13)throw std::runtime_error("Missing USD joint limits");
    for(size_t i=0;i<13;++i){if(!std::isfinite(lo[i]) || !std::isfinite(hi[i]) || lo[i]>=hi[i])throw std::runtime_error("Invalid USD joint limits");lower_[i]=lo[i];upper_[i]=hi[i];}
    encoder_=std::make_unique<nandu1::Network>(bundle+"/encoder.onnx",245,7);
    actor_=std::make_unique<nandu1::Network>(bundle+"/policy.onnx",profile_.actorWidth,13);
    command_.writeFromNonRT(Command{});health_.writeFromNonRT(Health{});requested_.writeFromNonRT(-1);
    const std::string topic=profile_.task=="Isaac-NANDU1-Blind-Flat-v1-2"?"/nandu1/command":"/nandu1/"+profile_.task.substr(profile_.task.find("v1-")+0)+"/command";
    std::string normalizedTopic=topic;std::replace(normalizedTopic.begin(),normalizedTopic.end(),'-','_');
    if(v14_)commandSub_=nh.subscribe<tron2_controllers::NANDU1V14Command>(normalizedTopic,1,[this](const tron2_controllers::NANDU1V14Command::ConstPtr& msg){
      Command v;v.v={msg->relative_heading_rate,msg->up3_vx_base_heading,msg->up3_height_world};v.received=ros::WallTime::now().toSec();
      v.invalid=!std::all_of(v.v.begin(),v.v.end(),[](double x){return std::isfinite(x);});command_.writeFromNonRT(v);
    });
    else if(profile_.kind==nandu1::Kind::TCP)commandSub_=nh.subscribe<tron2_controllers::NANDU1TCPCommand>(normalizedTopic,1,[this](const tron2_controllers::NANDU1TCPCommand::ConstPtr& msg){
      Command v;v.v={msg->relative_heading_rate,msg->tcp_vx_base_body,msg->tcp_height_world};v.received=ros::WallTime::now().toSec();
      v.invalid=!std::all_of(v.v.begin(),v.v.end(),[](double x){return std::isfinite(x);});command_.writeFromNonRT(v);
    });
    else commandSub_=nh.subscribe<tron2_controllers::NANDU1Command>(normalizedTopic,1,[this](const tron2_controllers::NANDU1Command::ConstPtr& msg){
      Command v;v.v={msg->yaw_rate_world_z,msg->up3_vx_base_body,msg->up3_height_world};v.received=ros::WallTime::now().toSec();
      v.invalid=!std::all_of(v.v.begin(),v.v.end(),[](double x){return std::isfinite(x);});
      command_.writeFromNonRT(v);
    });
    healthSub_=nh.subscribe<std_msgs::Float64MultiArray>("/nandu1/sensor_health",1,[this](const std_msgs::Float64MultiArray::ConstPtr& msg){
      if(msg->data.size()!=4 || !std::all_of(msg->data.begin(),msg->data.end(),[](double v){return std::isfinite(v)&&v>=0;}))return;
      Health h;std::copy_n(msg->data.begin(),3,h.age.begin());h.epoch=static_cast<uint64_t>(msg->data[3]);
      h.received=ros::WallTime::now().toSec();health_.writeFromNonRT(h);
    });
    modeSub_=nh.subscribe<std_msgs::String>("/nandu1/set_mode",1,[this](const std_msgs::String::ConstPtr& msg){
      if(msg->data=="POLICY")requested_.writeFromNonRT(1);
      else if(msg->data=="RESET")requested_.writeFromNonRT(2);
      else if(msg->data=="PASSIVE")requested_.writeFromNonRT(0);
    });
    trackingPub_=nh.advertise<std_msgs::Float64MultiArray>("/nandu1/tracking",1);
    resetEvidencePub_=nh.advertise<std_msgs::Float64MultiArray>("/nandu1/reset_evidence",1,true);
    diagnosticsPub_=nh.advertise<diagnostic_msgs::DiagnosticArray>("/nandu1/diagnostics",1,profile_.networkErrors);
    reset();return true;
  }catch(const std::exception& e){ROS_ERROR("NANDU1 init: %s",e.what());return false;}
}
void NANDU1Controller::reset(){
  ++resetGeneration_;resetEvidencePending_=true;
  runtime_.reset();qTarget_=nandu1::defaults();dqTarget_.fill(0);
  appliedCommand_={0,0,initialHeight_};std::fill(actions_.begin(),actions_.end(),0);
  processor_.reset(initialHeight_);smoothProcessor_.reset(initialHeight_);tcpProcessor_.reset(initialHeight_);
  externalApplied_=appliedCommand_;
  std::fill(estimate_.begin(),estimate_.end(),0);estimate_[6]=initialHeight_;
  initialized_=false;active_=false;fault_=false;faultReason_.clear();nextPolicy_=0;
  command_.writeFromNonRT(Command{});
}
void NANDU1Controller::starting(const ros::Time&){reset();}
void NANDU1Controller::stopping(const ros::Time&){active_=false;damping();if(profile_.networkErrors)stateDiagnostic();}
void NANDU1Controller::stateDiagnostic(){
  diagnostic_msgs::DiagnosticArray d;d.header.stamp=ros::Time::now();diagnostic_msgs::DiagnosticStatus s;
  s.name="nandu1/policy";s.level=fault_?2:0;s.message=fault_?faultReason_:(active_?"POLICY":"PASSIVE");
  for(const auto& entry:std::vector<std::pair<std::string,std::string>>{
      {"mode",fault_?"FAULT":(active_?"POLICY":"PASSIVE")},
      {"epoch",std::to_string(epoch_)},{"generation",std::to_string(resetGeneration_)}}){
    diagnostic_msgs::KeyValue v;v.key=entry.first;v.value=entry.second;s.values.push_back(v);
  }
  d.status.push_back(s);diagnosticsPub_.publish(d);
}
void NANDU1Controller::fault(const std::string& reason){
  if(!fault_)ROS_ERROR("NANDU1 FAULT: %s",reason.c_str());
  fault_=true;faultReason_=reason;active_=false;
  if(profile_.networkErrors){stateDiagnostic();return;}
  diagnostic_msgs::DiagnosticArray d;d.header.stamp=ros::Time::now();diagnostic_msgs::DiagnosticStatus s;
  s.name="nandu1/policy";s.level=2;s.message=reason;d.status.push_back(s);diagnosticsPub_.publish(d);
}
void NANDU1Controller::damping(){
  for(size_t i=0;i<joints_.size();++i)joints_[i].setCommand(joints_[i].getPosition(),0,0,nandu1::kd()[i],0,0);
}
void NANDU1Controller::update(const ros::Time&,const ros::Duration&){
  const double now=ros::WallTime::now().toSec();
  const Health h=*health_.readFromRT();int requested=*requested_.readFromRT();
  if(requested==2){reset();requested=1;requested_.writeFromNonRT(1);}
  if(h.epoch>0 && h.epoch!=epoch_){epoch_=h.epoch;reset();}
  // Explicit PASSIVE also disarms the watchdog. Active POLICY still latches
  // stale-sensor faults; intentional ROS/receiver shutdown must not do so.
  if(requested==0){active_=false;initialized_=false;damping();if(profile_.networkErrors)stateDiagnostic();return;}
  bool healthy=h.epoch>0 && now-h.received<.1;
  for(double age:h.age)healthy=healthy && age+(now-h.received)<.1;
  if(!healthy){if(initialized_)fault("Sensor state expired");damping();if(profile_.networkErrors)stateDiagnostic();return;}
  initialized_=true;
  if(requested==0 || (requested==-1 && !autoPolicy_)){active_=false;damping();if(profile_.networkErrors)stateDiagnostic();return;}
  if(fault_){damping();if(profile_.networkErrors)stateDiagnostic();return;}
  if(!active_){
    if(profile_.networkErrors){reset();initialized_=true;}
    runtime_.reset();processor_.reset(appliedCommand_[2]);smoothProcessor_.reset(appliedCommand_[2]);tcpProcessor_.reset(appliedCommand_[2]);externalApplied_=appliedCommand_;active_=true;nextPolicy_=now;ROS_INFO("NANDU1 sensors ready -> POLICY (%s)",profile_.task.c_str());
  }
  try{
    std::array<double,13> q,dq;
    for(size_t i=0;i<13;++i){q[i]=joints_[i].getPosition();dq[i]=joints_[i].getVelocity();
      if(!std::isfinite(q[i]) || !std::isfinite(dq[i]) || q[i]<lower_[i]-.05 || q[i]>upper_[i]+.05)throw std::runtime_error("Joint state outside limits");}
    if(now>=nextPolicy_){
      // Wall-time scheduler follows the real-time SDK bridge, never catches up with bursts.
      nextPolicy_+=.02;
      if(now-nextPolicy_>=.02)nextPolicy_=now+.02;
      auto quaternion=[](const hardware_interface::ImuSensorHandle& imu){auto p=imu.getOrientation();return Eigen::Quaterniond(p[3],p[0],p[1],p[2]);};
      auto gyro=[](const hardware_interface::ImuSensorHandle& imu){auto p=imu.getAngularVelocity();return Eigen::Vector3d(p[0],p[1],p[2]);};
      auto obs=runtime_.observe(quaternion(base_),gyro(base_),quaternion(up_),gyro(up_),q,dq);
      auto history=runtime_.append(obs);
      if(resetEvidencePending_){
        double lastInf=0,repeatError=0,cueInf=0,heading=0,target=0;
        for(int i=36;i<49;++i)lastInf=std::max(lastInf,std::abs(double(obs[i])));
        for(size_t i=0;i<history.size();++i)repeatError=std::max(repeatError,std::abs(double(history[i]-obs[i%49])));
        if(profile_.kind==nandu1::Kind::HeadingV141){
          heading=smoothProcessor_.heading();target=smoothProcessor_.target();
          for(float v:smoothProcessor_.state().cue)cueInf=std::max(cueInf,std::abs(double(v)));
        }else if(profile_.kind==nandu1::Kind::HeadingV14){heading=processor_.heading();target=processor_.target();}
        else if(profile_.kind==nandu1::Kind::TCP){heading=tcpProcessor_.heading();target=tcpProcessor_.target();}
        std_msgs::Float64MultiArray evidence;evidence.data={now,double(epoch_),double(resetGeneration_),lastInf,repeatError,heading,target,cueInf,
          appliedCommand_[0],appliedCommand_[1],appliedCommand_[2],initialHeight_};
        resetEvidencePub_.publish(evidence);resetEvidencePending_=false;
      }
      estimate_=encoder_->run(history);
      const Command command=*command_.readFromRT();if(command.invalid)throw std::runtime_error("Non-finite command");
      auto desired=command.v;
      if(command.received==0 || now-command.received>.5)desired={0,0,appliedCommand_[2]};
      for(int i=0;i<3;++i)desired[i]=std::clamp(desired[i],commandLower_[i],commandUpper_[i]);
      std::vector<float> commandFeatures,input;
      if(v14_){
        if(profile_.kind==nandu1::Kind::HeadingV141){
          smoothProcessor_.setExternal(desired);
          commandFeatures=smoothProcessor_.step(Eigen::Vector3f(obs[3],obs[4],obs[5]),gyro(base_).cast<float>(),.02f);
        }else commandFeatures=processor_.step(desired,Eigen::Vector3f(obs[3],obs[4],obs[5]),gyro(base_).cast<float>(),.02f);
        std::copy_n(commandFeatures.begin(),3,appliedCommand_.begin());
        input=nandu1::v14::actorInput(estimate_,obs,commandFeatures);
      }else{
        for(int i=0;i<3;++i)externalApplied_[i]+=std::clamp(desired[i]-externalApplied_[i],-commandRate_[i]*.02,commandRate_[i]*.02);
        if(profile_.kind==nandu1::Kind::TCP){
          commandFeatures=tcpProcessor_.step(externalApplied_,Eigen::Vector3f(obs[3],obs[4],obs[5]),gyro(base_).cast<float>(),.02f);
          std::copy_n(commandFeatures.begin(),3,appliedCommand_.begin());
        }else appliedCommand_=externalApplied_;
        input=estimate_;input.insert(input.end(),obs.begin(),obs.end());for(double value:appliedCommand_)input.push_back(value);
      }
      actions_=actor_->run(input);auto target=runtime_.decode(actions_);
      const size_t clipped=nandu1::clipPositionTargets(target.first,lower_,upper_);
      if(clipped)ROS_WARN_THROTTLE(2.0,"NANDU1: clipped %zu policy position targets to inclusive physical limits",clipped);
      qTarget_=target.first;dqTarget_=target.second;runtime_.applied(actions_);
      std_msgs::Float64MultiArray telemetry;
      telemetry.data={now,double(epoch_)};
      telemetry.data.insert(telemetry.data.end(),appliedCommand_.begin(),appliedCommand_.end());
      telemetry.data.insert(telemetry.data.end(),estimate_.begin(),estimate_.end());
      telemetry.data.insert(telemetry.data.end(),actions_.begin(),actions_.end());
      for(const auto& values:{q,dq,qTarget_,dqTarget_})telemetry.data.insert(telemetry.data.end(),values.begin(),values.end());
      for(const auto& joint:joints_)telemetry.data.push_back(joint.getEffort());
      for(size_t i=0;i<13;++i){double raw=nandu1::kp()[i]*(qTarget_[i]-q[i])+nandu1::kd()[i]*(dqTarget_[i]-dq[i]);telemetry.data.push_back(std::clamp(raw,-nandu1::effort()[i],nandu1::effort()[i]));telemetry.data.push_back(raw!=std::clamp(raw,-nandu1::effort()[i],nandu1::effort()[i]));}
      if(v14_){
        telemetry.data.insert(telemetry.data.end(),commandFeatures.begin()+3,commandFeatures.end());
        telemetry.data.insert(telemetry.data.end(),input.begin(),input.begin()+7);
        telemetry.data.push_back(profile_.kind==nandu1::Kind::HeadingV141?smoothProcessor_.heading():processor_.heading());telemetry.data.push_back(profile_.kind==nandu1::Kind::HeadingV141?smoothProcessor_.target():processor_.target());
        telemetry.data.insert(telemetry.data.end(),desired.begin(),desired.end());
      }else if(profile_.kind==nandu1::Kind::TCP){
        telemetry.data.push_back(tcpProcessor_.heading());telemetry.data.push_back(tcpProcessor_.target());
        telemetry.data.insert(telemetry.data.end(),desired.begin(),desired.end());
      }
      trackingPub_.publish(telemetry);
      if(profile_.networkErrors)stateDiagnostic();
      else{
        diagnostic_msgs::DiagnosticArray diag;diag.header.stamp=ros::Time::now();
        diagnostic_msgs::DiagnosticStatus status;status.name="nandu1/policy";status.level=0;status.message="POLICY";
        diag.status.push_back(status);diagnosticsPub_.publish(diag);
      }
    }
    for(size_t i=0;i<13;++i)joints_[i].setCommand(qTarget_[i],dqTarget_[i],nandu1::kp()[i],nandu1::kd()[i],0,0);
  }catch(const std::exception& e){fault(e.what());damping();}
  if(fault_){
    if(profile_.networkErrors)stateDiagnostic();
    else{diagnostic_msgs::DiagnosticArray d;d.header.stamp=ros::Time::now();diagnostic_msgs::DiagnosticStatus s;
      s.name="nandu1/policy";s.level=2;s.message=faultReason_;d.status.push_back(s);diagnosticsPub_.publish(d);}
  }
}
}
PLUGINLIB_EXPORT_CLASS(tron2_controller::NANDU1Controller,controller_interface::ControllerBase)
