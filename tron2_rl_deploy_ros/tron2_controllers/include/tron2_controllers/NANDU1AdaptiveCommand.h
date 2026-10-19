#pragma once
#include "NANDU1Runtime.h"
namespace nandu1::adaptive {
inline float bound(float x,float a,float b){return std::clamp(x,a,b);}
inline float wrap(float x){return std::atan2(std::sin(x),std::cos(x));}
inline float quiet(float x,float low,float high){float t=bound((std::abs(x)-low)/(high-low),0.f,1.f);return 1-t*t*(3-2*t);}
inline constexpr float DT=.02f;
inline constexpr float YAW_LIMITS[5]={.2f,.5f,.85f,1.2f,1.57f};
inline constexpr float VX_LIMITS[5]={.08f,.2f,.4f,.7f,1.f};
inline constexpr float HEIGHT_LOW[5]={.78f,.68f,.58f,.48f,.4f};
inline constexpr float HEIGHT_HIGH[5]={1.04f,1.12f,1.22f,1.32f,1.4f};
inline constexpr float HEIGHT_RATES[5]={.08f,.15f,.25f,.38f,.5f};
inline constexpr float BALANCE_HEIGHT_RATES[5]={.05f,.10f,.20f,.35f,.5f};
struct ImuProjection {float headingRate,worldYaw,tilt,norm;};
inline ImuProjection project(const std::vector<float>& obs){
  Eigen::Vector3f gravity(obs[3],obs[4],obs[5]),gyro(obs[0]/.25f,obs[1]/.25f,obs[2]/.25f);
  float norm=gravity.norm();Eigen::Vector3f up=norm>1e-6f?Eigen::Vector3f(-gravity/std::max(norm,1e-6f)):Eigen::Vector3f(0,0,1);
  Eigen::Vector3f ex(1,0,0),ey(0,1,0),forward=ex-up.x()*up;
  if(forward.norm()<=1e-6f)forward=ey-up.y()*up;
  forward/=std::max(forward.norm(),1e-6f);Eigen::Vector3f left=up.cross(forward);
  float projection=std::sqrt(std::max(1-up.x()*up.x(),1e-4f)),world=gyro.dot(up);
  return {projection>.01f?gyro.cross(ex).dot(left)/projection:world,world,std::acos(bound(up.z(),-1.f,1.f)),norm};
}
class CommandProcessor {
  float yawIntent_=0;bool turning_=false;
public:
  std::vector<float> command=std::vector<float>(3,0);
  float heightRate=0,heading=0,target=0;
  void reset(float height){command={0,0,height};heightRate=heading=target=yawIntent_=0;turning_=false;}
  void step(const std::vector<float>& goal,const ImuProjection& imu,int stage,bool fasterHeight=true){
    const float limit=YAW_LIMITS[stage];heading=wrap(heading+DT*imu.headingRate);
    float intent=bound(goal[0],-limit,limit);
    yawIntent_+=bound(DT/(.10f+DT)*(intent-yawIntent_),-2*DT,2*DT);
    bool turning=std::abs(intent)>1e-5f;if(turning_&&!turning)target=heading;
    float ff=turning?yawIntent_:0;target=wrap(target+ff*DT);
    float coupling=imu.worldYaw-imu.headingRate,raw=ff+1.5f*wrap(target-heading)+coupling;
    float desired=bound(raw,-limit,limit);
    if(turning&&raw!=desired)target=wrap(heading+(desired-coupling-ff)/1.5f);
    command[0]+=bound(desired-command[0],-2*DT,2*DT);
    command[1]+=bound(DT/(.15f+DT)*(goal[1]-command[1]),-.6f*DT,.6f*DT);
    float old=command[2],error=goal[2]-old,rate=fasterHeight?HEIGHT_RATES[stage]:BALANCE_HEIGHT_RATES[stage];
    command[2]=(std::abs(error)<=1e-4f&&std::abs(error)<=rate*DT)?goal[2]:old+bound(DT/(.15f+DT)*error,-rate*DT,rate*DT);
    heightRate=(command[2]-old)/DT;turning_=turning;
  }
};
}
