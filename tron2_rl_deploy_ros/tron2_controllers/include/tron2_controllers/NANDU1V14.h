#pragma once
#include "NANDU1Runtime.h"

namespace nandu1 {
namespace v14 {
inline std::array<Eigen::Vector3f,3> axes(const Eigen::Vector3f& gravity){
  if(!gravity.allFinite())throw std::runtime_error("Invalid projected gravity");
  Eigen::Vector3f up=gravity.norm()>1e-6f?Eigen::Vector3f(-gravity/gravity.norm()):Eigen::Vector3f(0,0,1);
  Eigen::Vector3f forward=Eigen::Vector3f::UnitX()-up.x()*up;
  if(forward.norm()<=1e-6f)forward=Eigen::Vector3f::UnitY()-up.y()*up;
  forward/=std::max(forward.norm(),1e-6f);
  return {forward,up.cross(forward),up};
}
inline float wrap(float angle){return std::atan2(std::sin(angle),std::cos(angle));}
inline std::vector<float> actorInput(const std::vector<float>& estimated,const std::vector<float>& obs,const std::vector<float>& command){
  finite(estimated,7);finite(obs,49);finite(command,6);
  const auto basis=axes(Eigen::Vector3f(obs[3],obs[4],obs[5]));
  std::vector<float> out;
  for(int offset:{0,3})for(const auto& axis:basis)out.push_back(axis.dot(Eigen::Vector3f(estimated[offset],estimated[offset+1],estimated[offset+2])));
  out.push_back(estimated[6]);out.insert(out.end(),obs.begin(),obs.end());out.insert(out.end(),command.begin(),command.end());
  finite(out,62);return out;
}
class CommandProcessor {
  int stage_=4;
  float heading_=0,target_=0;
  bool turning_=false;
  std::array<float,3> applied_{0,0,.9f};
public:
  void configure(int stage,float height){if(stage<0||stage>4)throw std::runtime_error("Invalid v1.4 stage");stage_=stage;reset(height);}
  void reset(float height){if(!std::isfinite(height))throw std::runtime_error("Invalid initial height");heading_=target_=0;turning_=false;applied_={0,0,std::clamp(height,.4f,1.4f)};}
  float heading()const{return heading_;}float target()const{return target_;}
  std::vector<float> step(const std::array<double,3>& external,const Eigen::Vector3f& gravity,const Eigen::Vector3f& gyro,float dt=.02f){
    if(!gyro.allFinite()||!std::all_of(external.begin(),external.end(),[](double x){return std::isfinite(x);})||!std::isfinite(dt)||dt<1e-6f||dt>.1f)
      throw std::runtime_error("Invalid v1.4 command processor input");
    static constexpr float yawLimits[5]={.2f,.5f,.85f,1.2f,1.57f},vxLimits[5]={.08f,.2f,.4f,.7f,1.f};
    static constexpr float lower[5]={.78f,.68f,.58f,.48f,.4f},upper[5]={1.04f,1.12f,1.22f,1.32f,1.4f},rate[5]={.05f,.1f,.2f,.35f,.5f};
    const float yl=yawLimits[stage_],vl=vxLimits[stage_],lo=lower[stage_],hi=upper[stage_],hr=rate[stage_];
    std::array<float,3> goal{std::clamp(float(external[0]),-yl,yl),std::clamp(float(external[1]),-vl,vl),std::clamp(float(external[2]),lo,hi)};
    const auto previous=applied_;const auto basis=axes(gravity);const auto& left=basis[1];const auto& up=basis[2];
    const float projection=std::sqrt(std::max(1.f-up.x()*up.x(),1e-4f));
    float headingRate=gyro.cross(Eigen::Vector3f::UnitX()).dot(left)/projection;
    if(projection<=.01f)headingRate=gyro.dot(up);
    heading_=wrap(heading_+headingRate*dt);
    bool turning=std::abs(goal[0])>1e-5f;
    if(turning_&&!turning)target_=heading_;
    target_=wrap(target_+(turning?goal[0]*dt:0.f));
    const float coupling=gyro.dot(up)-headingRate;
    const float raw=goal[0]+1.5f*wrap(target_-heading_)+coupling;
    const float yaw=std::clamp(raw,-yl,yl);
    if(raw!=yaw)target_=wrap(heading_+(yaw-coupling-goal[0])/1.5f);
    turning_=turning;
    float height=(previous[2]<lo||previous[2]>hi)?goal[2]:previous[2]+std::clamp(goal[2]-previous[2],-hr*dt,hr*dt);
    applied_={yaw,goal[1],std::clamp(height,lo,hi)};
    std::vector<float> result(applied_.begin(),applied_.end());const float scale[3]={yl,vl,hr};
    for(int i=0;i<3;++i)result.push_back(std::clamp((applied_[i]-previous[i])/dt/scale[i],-1.f,1.f));
    finite(result,6);return result;
  }
};
}
}
