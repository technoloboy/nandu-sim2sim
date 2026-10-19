#pragma once
#include "NANDU1AdaptiveCommand.h"
namespace nandu1::adaptive {
class SlowMemory {
  std::array<float,2> integral_{};
public:
  std::vector<float> filter=std::vector<float>(2,0),normalized=std::vector<float>(2,0);
  void reset(){integral_.fill(0);filter.assign(2,0);normalized.assign(2,0);}
  void step(const std::vector<float>& command,const std::vector<float>& encoded,const std::vector<float>& intent,
      float hdot,const ImuProjection& imu,bool clipped){
    if(clipped||imu.norm<=1e-6f||imu.tilt>float(M_PI)/4)return;
    float errors[2]={command[1]-encoded[3],command[0]-imu.worldYaw};
    const float caps[2]={.3f,.6f},limits[2]={.2f,.4f},scale[2]={.1f,.2f};
    float gates[2]={quiet(intent[1],.02f,.08f),quiet(intent[0],.03f,.10f)};
    for(int i=0;i<2;++i){
      filter[i]+=DT/(.25f+DT)*(bound(errors[i],-caps[i],caps[i])-filter[i]);
      float g=gates[i]*quiet(hdot,.01f,.04f);
      integral_[i]=bound(integral_[i]+DT*(g*filter[i]-(1-g)*integral_[i]),-limits[i],limits[i]);
      normalized[i]=integral_[i]/scale[i];
    }
  }
};
class HeightMemory {
  float integral_=0;
public:
  float filter=0,normalized=0;
  void reset(){integral_=filter=normalized=0;}
  void step(float error,float hdot,const ImuProjection& imu,bool clipped){
    if(clipped||imu.norm<=1e-6f||imu.tilt>float(M_PI)/4)return;
    filter+=DT/(.5f+DT)*(bound(error,-.1f,.1f)-filter);
    float g=quiet(hdot,.005f,.02f),leak=g/10+(1-g);
    integral_=bound(integral_+DT*(g*filter-leak*integral_),-.1f,.1f);normalized=integral_/.05f;
  }
};
}
