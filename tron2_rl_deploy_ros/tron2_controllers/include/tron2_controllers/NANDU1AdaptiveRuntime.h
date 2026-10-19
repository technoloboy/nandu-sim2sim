#pragma once
#include "NANDU1AdaptiveOnnx.h"
#include "NANDU1AdaptiveMemory.h"
namespace nandu1::adaptive {
struct Result {
  Vec observation,history,commands,encoded,dynamic,actorInput,action,hidden,eta,memoryFilter,heightFilter;
  float heightRate=0,heading=0,headingTarget=0;
};
class Runtime {
  nandu1::Runtime history_;
  Vec previous_=Vec(13,0),hidden_=Vec(64,0),eta_=Vec(8,0);
  int stage_=4;int64_t lastId_=-1;bool clipped_=false,targetClipped_=false;
  Result cached_;
  bool heightMemoryEnabled_=true,fasterHeight_=true;
  int width_=71;
  Result advance(Vec obs,Vec intent,int64_t id,Networks& n){
    finite(obs,49);finite(intent,3);std::copy(previous_.begin(),previous_.end(),obs.begin()+36);
    intent[0]=bound(intent[0],-YAW_LIMITS[stage_],YAW_LIMITS[stage_]);
    intent[1]=bound(intent[1],-VX_LIMITS[stage_],VX_LIMITS[stage_]);
    intent[2]=bound(intent[2],HEIGHT_LOW[stage_],HEIGHT_HIGH[stage_]);
    auto imu=project(obs);processor.step(intent,imu,stage_,fasterHeight_);
    Result r;r.observation=obs;r.history=history_.append(obs);r.encoded=n.encode(r.history);
    r.dynamic=Vec(width_-59,0);
    if(lastId_>=0){
      auto ctx=n.context(obs,previous_,hidden_,eta_);hidden_=ctx.hidden;eta_=ctx.filtered;
      memory.step(processor.command,r.encoded,intent,processor.heightRate,imu,clipped_);
      if(heightMemoryEnabled_)heightMemory.step(processor.command[2]-r.encoded[6],processor.heightRate,imu,clipped_);
      r.dynamic=memory.normalized;r.dynamic.insert(r.dynamic.end(),eta_.begin(),eta_.end());
      r.dynamic.push_back(processor.heightRate/.5f);
      if(width_==71)r.dynamic.push_back(heightMemoryEnabled_?heightMemory.normalized:0.f);
    }
    r.commands=processor.command;r.actorInput=r.encoded;
    r.actorInput.insert(r.actorInput.end(),obs.begin(),obs.end());
    r.actorInput.insert(r.actorInput.end(),r.commands.begin(),r.commands.end());
    r.actorInput.insert(r.actorInput.end(),r.dynamic.begin(),r.dynamic.end());finite(r.actorInput,width_);
    if(n.rawWidth()!=int(r.actorInput.size()))throw std::runtime_error("Adaptive runtime/network profile mismatch");
    r.action=n.act(r.actorInput);previous_=r.action;r.hidden=hidden_;r.eta=eta_;
    r.memoryFilter=memory.filter;if(width_==71)r.heightFilter={heightMemoryEnabled_?heightMemory.filter:0.f};
    r.heightRate=processor.heightRate;r.heading=processor.heading;r.headingTarget=processor.target;
    history_.applied(r.action);lastId_=id;cached_=r;return r;
  }
public:
  CommandProcessor processor;SlowMemory memory;HeightMemory heightMemory;
  explicit Runtime(bool heightMemory=true):Runtime(heightMemory?71:70,heightMemory,heightMemory){}
  Runtime(int width,bool heightMemory,bool fasterHeight):heightMemoryEnabled_(heightMemory),fasterHeight_(fasterHeight),width_(width){
    if((width!=70&&width!=71)||(width==70&&heightMemory))throw std::runtime_error("Invalid adaptive runtime profile");
  }
  void reset(float height,int stage=4){
    if(!std::isfinite(height)||stage<0||stage>4)throw std::runtime_error("Invalid adaptive reset");
    history_.reset();previous_.assign(13,0);hidden_.assign(64,0);eta_.assign(8,0);
    processor.reset(height);memory.reset();heightMemory.reset();stage_=stage;lastId_=-1;cached_=Result{};
    clipped_=targetClipped_=false;
  }
  void reportIntervalClipping(bool action,bool target){clipped_=action;targetClipped_=target;}
  Result step(const Vec& obs,const Vec& intent,int64_t id,Networks& networks){
    if(id<0||id<lastId_)throw std::runtime_error("Adaptive step ID moves backwards");
    if(id==lastId_)return cached_;
    // Commit temporal state only after all three graph calls succeed.
    auto next=*this;auto result=next.advance(obs,intent,id,networks);*this=std::move(next);return result;
  }
};
}
