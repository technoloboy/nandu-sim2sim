#pragma once
// Portable reference only. No ROS, network, or robot activation.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace nandu1 { namespace v141 {
inline float wrap(float x){return std::atan2(std::sin(x),std::cos(x));}
inline std::array<Eigen::Vector3f,3> axes(const Eigen::Vector3f& g){
  if(!g.allFinite())throw std::runtime_error("Invalid gravity");
  Eigen::Vector3f up=g.norm()>1e-6f?Eigen::Vector3f(-g/g.norm()):Eigen::Vector3f(0,0,1);
  Eigen::Vector3f forward=Eigen::Vector3f::UnitX()-up.x()*up;
  if(forward.norm()<=1e-6f)forward=Eigen::Vector3f::UnitY()-up.y()*up;
  forward/=std::max(forward.norm(),1e-6f);
  return {forward,up.cross(forward),up};
}
inline std::vector<float> actorInput(const std::vector<float>& enc,const std::vector<float>& obs,const std::vector<float>& cmd){
  if(enc.size()!=7||obs.size()!=49||cmd.size()!=6)throw std::runtime_error("Actor input dimensions");
  for(const auto* values:{&enc,&obs,&cmd})for(float x:*values)if(!std::isfinite(x))throw std::runtime_error("Non-finite actor input");
  auto basis=axes(Eigen::Vector3f(obs[3],obs[4],obs[5]));std::vector<float> out;
  for(int offset:{0,3})for(const auto& axis:basis)out.push_back(axis.dot(Eigen::Vector3f(enc[offset],enc[offset+1],enc[offset+2])));
  out.push_back(enc[6]);out.insert(out.end(),obs.begin(),obs.end());out.insert(out.end(),cmd.begin(),cmd.end());return out;
}
class CommandProcessor {
public:
  struct State {int stage=4;float heading=0,target=0;bool turning=false,pending=false,is_step=false;
    float height_rate=0;std::array<float,3> applied{0,0,.9f},goal{0,0,.9f},cue{0,0,0};long long tick=-1;};
  static constexpr const char* schema="nandu1_v1_4_1_command_processor_v1";
private:
  State s;
public:
  void setStage(int stage){if(stage<0||stage>4)throw std::runtime_error("Invalid stage");s.stage=stage;}
  void configure(int stage,float height){setStage(stage);reset(height);}
  void reset(float height){if(!std::isfinite(height))throw std::runtime_error("Invalid height");int stage=s.stage;s=State{};s.stage=stage;s.applied={0,0,std::clamp(height,.4f,1.4f)};s.goal=s.applied;}
  float heading()const{return s.heading;}float target()const{return s.target;}
  State state()const{return s;}
  void restore(const State& state,const std::string& version){
    if(version!=schema)throw std::runtime_error("Processor schema mismatch");
    if(state.stage<0||state.stage>4||!std::isfinite(state.heading)||!std::isfinite(state.target)||!std::isfinite(state.height_rate))throw std::runtime_error("Invalid state");
    for(const auto* a:{&state.applied,&state.goal,&state.cue})for(float x:*a)if(!std::isfinite(x))throw std::runtime_error("Invalid state");s=state;
  }
  void setExternal(const std::array<double,3>& external,bool finite_step=false){
    for(double x:external)if(!std::isfinite(x))throw std::runtime_error("Invalid command");
    static constexpr float yaw[5]={.2f,.5f,.85f,1.2f,1.57f},vx[5]={.08f,.2f,.4f,.7f,1.f};
    static constexpr float lower[5]={.78f,.68f,.58f,.48f,.4f},upper[5]={1.04f,1.12f,1.22f,1.32f,1.4f};
    s.goal={std::clamp(float(external[0]),-yaw[s.stage],yaw[s.stage]),
            std::clamp(float(external[1]),-vx[s.stage],vx[s.stage]),
            std::clamp(float(external[2]),lower[s.stage],upper[s.stage])};s.pending=finite_step;
  }
  std::vector<float> step(const Eigen::Vector3f& gravity,const Eigen::Vector3f& gyro,float dt=.02f,long long tick=-1){
    if(!gyro.allFinite()||!std::isfinite(dt)||dt<1e-6f||dt>.1f)throw std::runtime_error("Invalid inputs");
    if(tick>=0&&tick==s.tick)return output();
    static constexpr float yaw[5]={.2f,.5f,.85f,1.2f,1.57f},vx[5]={.08f,.2f,.4f,.7f,1.f};
    static constexpr float lower[5]={.78f,.68f,.58f,.48f,.4f},upper[5]={1.04f,1.12f,1.22f,1.32f,1.4f},rate[5]={.05f,.1f,.2f,.35f,.5f};
    float yl=yaw[s.stage],vl=vx[s.stage],lo=lower[s.stage],hi=upper[s.stage],hr=rate[s.stage];
    auto goal=s.goal;goal[0]=std::clamp(goal[0],-yl,yl);goal[1]=std::clamp(goal[1],-vl,vl);goal[2]=std::clamp(goal[2],lo,hi);
    auto previous=s.applied;auto basis=axes(gravity);auto up=basis[2];float projection=std::sqrt(std::max(1.f-up.x()*up.x(),1e-4f));
    float headingRate=projection>.01f?gyro.cross(Eigen::Vector3f::UnitX()).dot(basis[1])/projection:gyro.dot(up);
    s.heading=wrap(s.heading+headingRate*dt);bool turning=std::abs(goal[0])>1e-5f;
    if(s.turning&&!turning)s.target=s.heading;
    s.target=wrap(s.target+(turning?goal[0]*dt:0));float coupling=gyro.dot(up)-headingRate;
    float raw=goal[0]+1.5f*wrap(s.target-s.heading)+coupling;float appliedYaw=std::clamp(raw,-yl,yl);
    if(raw!=appliedYaw)s.target=wrap(s.heading+(appliedYaw-coupling-goal[0])/1.5f);
    s.turning=turning;s.is_step=s.pending||previous[2]<lo||previous[2]>hi;
    float height=s.is_step?goal[2]:previous[2]+std::clamp(goal[2]-previous[2],-hr*dt,hr*dt);
    s.applied={appliedYaw,goal[1],std::clamp(height,lo,hi)};
    s.height_rate=s.is_step?0:(s.applied[2]-previous[2])/dt;float alpha=dt/(.08f+dt);
    s.cue[0]=(1-alpha)*s.cue[0]+alpha*std::tanh((s.applied[0]-previous[0])/dt/10.f);
    s.cue[1]=(1-alpha)*s.cue[1]+alpha*std::tanh((s.applied[1]-previous[1])/dt/5.f);
    s.cue[2]=std::clamp(s.height_rate/.5f,-1.f,1.f);s.pending=false;s.tick=tick;return output();
  }
  std::vector<float> output()const{std::vector<float> out(s.applied.begin(),s.applied.end());out.insert(out.end(),s.cue.begin(),s.cue.end());return out;}
};
}}
