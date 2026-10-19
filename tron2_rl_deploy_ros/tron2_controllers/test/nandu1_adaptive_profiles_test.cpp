#include "tron2_controllers/NANDU1AdaptiveProfiles.h"
#include <cassert>
using namespace nandu1::adaptive;
int main(){
  auto b=profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-3");
  assert(b.width==71&&!b.heightMemory&&!b.fasterHeight&&b.stage==1);
  assert(profileFor("Isaac-NANDU1-Blind-Flat-Play-v1-3-1-2-3").canonical==b.canonical);
  auto d=profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-5");
  assert(d.width==71&&d.heightMemory&&!d.fasterHeight&&d.stage==4);
  auto f=profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-7");
  assert(f.width==71&&f.heightMemory&&!f.fasterHeight&&f.stage==2);
  auto a=profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-2");
  assert(a.width==70&&!a.heightMemory&&!a.fasterHeight&&a.stage==4);
  auto e=profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-6");
  assert(e.width==71&&e.heightMemory&&e.fasterHeight&&e.stage==4);
  bool rejected=false;try{profileFor("Isaac-NANDU1-Blind-Flat-v1-3-1-2-4");}catch(const std::exception&){rejected=true;}
  assert(rejected);return 0;
}
