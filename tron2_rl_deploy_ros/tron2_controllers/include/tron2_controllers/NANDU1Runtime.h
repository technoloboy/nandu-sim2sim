#pragma once
#include <Eigen/Geometry>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nandu1 {
inline std::array<double,13> defaults(){return {-.475,0,0,.95,0,.475,0,0,.95,0,-.15,-1.78,1.93};}
inline const std::array<int,11>& positionIds(){static const std::array<int,11> v{0,1,2,3,5,6,7,8,10,11,12};return v;}
inline const std::array<double,13>& kp(){static const std::array<double,13> v{159.67,159.67,53.22,159.67,0,159.67,159.67,53.22,159.67,0,480,240,160};return v;}
inline const std::array<double,13>& kd(){static const std::array<double,13> v{10.16,10.16,3.39,10.16,.6,10.16,10.16,3.39,10.16,.6,50,25,17};return v;}
inline const std::array<double,13>& effort(){static const std::array<double,13> v{140,140,40,140,20,140,140,40,140,20,181,136,136};return v;}
inline std::vector<std::string> joints(){return {"proximal_pitch_L_Joint","proximal_roll_L_Joint","proximal_yaw_L_Joint","knee_L_Joint","wheel_L_Joint","proximal_pitch_R_Joint","proximal_roll_R_Joint","proximal_yaw_R_Joint","knee_R_Joint","wheel_R_Joint","arm1","arm2","arm3"};}
inline void finite(const std::vector<float>& v,size_t n){
  if(v.size()!=n || !std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Invalid NANDU1 vector");
}
// Project decoded position targets onto inclusive, asset-authoritative limits.
// Keep raw policy actions/history and the two wheel velocity targets unchanged.
inline size_t clipPositionTargets(std::array<double,13>& q,const std::array<double,13>& lower,
    const std::array<double,13>& upper){
  for(size_t i=0;i<13;++i)
    if(!std::isfinite(q[i]) || !std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>=upper[i])
      throw std::runtime_error("Invalid NANDU1 position target or limits");
  size_t clipped=0;
  for(int i:positionIds()){
    const double bounded=std::clamp(q[i],lower[i],upper[i]);
    clipped+=bounded!=q[i];q[i]=bounded;
  }
  return clipped;
}
class Runtime {
  std::vector<float> history_,last_=std::vector<float>(13,0);
public:
  void reset(){history_.clear();std::fill(last_.begin(),last_.end(),0);}
  std::vector<float> observe(Eigen::Quaterniond base,const Eigen::Vector3d& bg,Eigen::Quaterniond up,const Eigen::Vector3d& ug,
      const std::array<double,13>& q,const std::array<double,13>& dq){
    if(!base.coeffs().allFinite() || !up.coeffs().allFinite() || base.norm()<1e-6 || up.norm()<1e-6)throw std::runtime_error("Invalid IMU quaternion");
    std::vector<float> o;
    for(const auto& v:{Eigen::Vector3d(bg*.25),Eigen::Vector3d(base.normalized().conjugate()*Eigen::Vector3d(0,0,-1)),
        Eigen::Vector3d(ug*.25),Eigen::Vector3d(up.normalized().conjugate()*Eigen::Vector3d(0,0,-1))})
      for(int i=0;i<3;++i)o.push_back(v[i]);
    auto d=defaults();for(int i:positionIds())o.push_back(q[i]-d[i]);
    for(double v:dq)o.push_back(v*.05);
    o.insert(o.end(),last_.begin(),last_.end());finite(o,49);return o;
  }
  std::vector<float> append(const std::vector<float>& o){
    finite(o,49);
    if(history_.empty()){for(int i=0;i<5;++i)history_.insert(history_.end(),o.begin(),o.end());}
    else{history_.erase(history_.begin(),history_.begin()+49);history_.insert(history_.end(),o.begin(),o.end());}
    return history_;
  }
  std::pair<std::array<double,13>,std::array<double,13>> decode(const std::vector<float>& a){
    finite(a,13);auto q=defaults();std::array<double,13> dq{};
    for(int i=0;i<11;++i)q[positionIds()[i]]+=.25*a[i];
    dq[4]=a[11];dq[9]=a[12];return {q,dq};
  }
  void applied(const std::vector<float>& a){finite(a,13);last_=a;}
};
}
