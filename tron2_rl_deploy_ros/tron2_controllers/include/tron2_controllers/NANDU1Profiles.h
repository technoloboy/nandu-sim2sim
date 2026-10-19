#pragma once
#include "NANDU1V14.h"
#include <string>
namespace nandu1 {
enum class Kind {Body,TCP,HeadingV14,HeadingV141};
struct Profile {std::string task;Kind kind;int actorWidth;bool networkErrors=false;};
inline Profile selectProfile(std::string task){
  const std::string alias="Flat-Play-v";auto at=task.find(alias);
  if(at!=std::string::npos)task.replace(at,alias.size(),"Flat-v");
  if(task=="Isaac-NANDU1-Blind-Flat-v1-3-1-2")return {task,Kind::TCP,59,true};
  for(const std::string version:{"1","2","3","2-1","3-1","4","4-1"}){
    if(task=="Isaac-NANDU1-Blind-Flat-v1-"+version){
      Kind kind=version=="3-1"?Kind::TCP:version=="4"?Kind::HeadingV14:version=="4-1"?Kind::HeadingV141:Kind::Body;
      return {task,kind,(kind==Kind::HeadingV14||kind==Kind::HeadingV141)?62:59};
    }
  }
  throw std::runtime_error("Unsupported NANDU1 task/Play alias: "+task);
}
namespace v131 {
// Identical sensory heading hold to training; no world pose or TCP truth input.
class CommandProcessor {
  int stage_=4;float heading_=0,target_=0;bool turning_=false;
public:
  void configure(int stage,float height){if(stage<0||stage>4||!std::isfinite(height))throw std::runtime_error("Invalid TCP processor configuration");stage_=stage;reset(height);}
  void reset(float){heading_=target_=0;turning_=false;}
  float heading()const{return heading_;}float target()const{return target_;}
  std::vector<float> step(const std::array<double,3>& external,const Eigen::Vector3f& gravity,const Eigen::Vector3f& gyro,float dt){
    if(!gravity.allFinite()||!gyro.allFinite()||!std::isfinite(dt)||dt<1e-6f||dt>.1f)throw std::runtime_error("Invalid TCP processor inputs");
    for(double x:external)if(!std::isfinite(x))throw std::runtime_error("Non-finite TCP command");
    const float limits[5]={.2f,.5f,.85f,1.2f,1.57f};float limit=limits[stage_];
    auto basis=v14::axes(gravity);auto up=basis[2];float projection=std::sqrt(std::max(1.f-up.x()*up.x(),1e-4f));
    float rate=projection>.01f?gyro.cross(Eigen::Vector3f::UnitX()).dot(basis[1])/projection:gyro.dot(up);
    heading_=v14::wrap(heading_+rate*dt);float intent=std::clamp(float(external[0]),-limit,limit);bool turning=std::abs(intent)>1e-5f;
    if(turning_&&!turning)target_=heading_;
    target_=v14::wrap(target_+(turning?intent*dt:0));float coupling=gyro.dot(up)-rate;
    float raw=intent+1.5f*v14::wrap(target_-heading_)+coupling;float yaw=std::clamp(raw,-limit,limit);
    if(raw!=yaw)target_=v14::wrap(heading_+(yaw-coupling-intent)/1.5f);
    turning_=turning;return {yaw,float(external[1]),float(external[2])};
  }
}
;
}
}
