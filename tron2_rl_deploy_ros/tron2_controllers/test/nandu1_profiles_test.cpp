#include "tron2_controllers/NANDU1Profiles.h"
#include <cassert>
#include <iostream>
int main(){
  using namespace nandu1;
  auto p=selectProfile("Isaac-NANDU1-Blind-Flat-Play-v1-2-1");
  assert(p.task=="Isaac-NANDU1-Blind-Flat-v1-2-1" && p.kind==Kind::Body && p.actorWidth==59);
  assert(selectProfile("Isaac-NANDU1-Blind-Flat-v1-3-1").kind==Kind::TCP);
  assert(selectProfile("Isaac-NANDU1-Blind-Flat-v1-4-1").actorWidth==62);
  auto e=selectProfile("Isaac-NANDU1-Blind-Flat-Play-v1-3-1-2");
  assert(e.kind==Kind::TCP && e.actorWidth==59);
  bool rejected=false;try{selectProfile("Isaac-NANDU1-Blind-Flat-v1-3-2");}catch(const std::exception&){rejected=true;}assert(rejected);
  v131::CommandProcessor processor;processor.configure(4,.9);
  auto first=processor.step({0,0,1.2},{0,0,-1},{0,0,.2},.02);
  assert(std::abs(first[0]+.006f)<1e-6f && first[1]==0 && first[2]==1.2f);
  processor.reset(.9);
  assert(processor.heading()==0 && processor.target()==0);
  assert(processor.step({0,0,.9},{0,0,-1},{0,0,0},.02)[0]==0);
  rejected=false;try{processor.step({0,0,.9},{0,0,-1},{0,0,0},0);}catch(const std::exception&){rejected=true;}assert(rejected);
  std::cout<<"NANDU1 profiles and sensory TCP yaw processor passed\n";
}
