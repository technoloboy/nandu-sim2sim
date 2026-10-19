#include "tron2_controllers/NANDU1Runtime.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
  nandu1::Runtime r;
  auto q=nandu1::defaults();
  std::array<double,13> dq{};
  Eigen::Quaterniond imu(1,0,0,0);
  auto obs=r.observe(imu,{4,8,12},imu,{0,4,0},q,dq);
  assert(obs.size()==49 && obs[0]==1 && obs[1]==2 && obs[2]==3);
  assert(obs[5]==-1 && obs[11]==-1);
  auto h=r.append(obs);
  assert(h.size()==245 && h[0]==1 && h[196]==1);
  obs[0]=7;h=r.append(obs);
  assert(h[0]==1 && h[196]==7);
  std::vector<float> a(13);
  for(int i=0;i<13;++i)a[i]=i;
  auto target=r.decode(a);
  assert(target.second[4]==11 && target.second[9]==12);
  assert(std::abs(target.first[10]-1.85)<1e-6);
  assert(std::abs(target.first[5]-1.475)<1e-6);
  // SDK order limits are applied AFTER decoding the policy-order actions.
  auto lower=nandu1::defaults(),upper=lower;
  for(int i=0;i<13;++i){lower[i]-=.5;upper[i]+=.5;}
  for(float sign:{-1.f,1.f}){
    std::vector<float> large(13,sign*4);
    large[11]=17;large[12]=-19;
    auto bounded=r.decode(large);
    assert(nandu1::clipPositionTargets(bounded.first,lower,upper)==11);
    for(int i:nandu1::positionIds())assert(bounded.first[i]==(sign<0?lower[i]:upper[i]));
    assert(bounded.second[4]==17 && bounded.second[9]==-19);
    assert(bounded.first[4]==0 && bounded.first[9]==0);
    r.applied(large);
    auto next=r.observe(imu,{0,0,0},imu,{0,0,0},q,dq);
    for(int i=0;i<13;++i)assert(next[36+i]==large[i]);
  }
  // Inclusive boundaries, interior targets, and wheel position placeholders
  // are unchanged: no soft margin or hidden wheel velocity clipping.
  for(const auto& endpoint:{lower,upper}){
    auto atLimit=endpoint;
    assert(nandu1::clipPositionTargets(atLimit,lower,upper)==0);
    assert(atLimit==endpoint);
  }
  auto interior=nandu1::defaults();interior[4]=100;interior[9]=-100;
  auto original=interior;
  assert(nandu1::clipPositionTargets(interior,lower,upper)==0 && interior==original);
  auto invalid=lower;invalid[10]=NAN;bool invalidThrew=false;
  try{nandu1::clipPositionTargets(invalid,lower,upper);}catch(const std::exception&){invalidThrew=true;}
  assert(invalidThrew);
  auto badLower=lower;badLower[12]=upper[12]+1;invalidThrew=false;
  try{nandu1::clipPositionTargets(interior,badLower,upper);}catch(const std::exception&){invalidThrew=true;}
  assert(invalidThrew);
  r.reset();obs[0]=9;h=r.append(obs);assert(h[0]==9);
  a[0]=NAN;bool threw=false;
  try{r.decode(a);}catch(const std::exception&){threw=true;}
  assert(threw);
  std::cout<<"NANDU1 runtime tests passed\n";
}
