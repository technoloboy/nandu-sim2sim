#pragma once
#include <array>
#include <stdexcept>
#include <string>
namespace nandu1::adaptive {
struct Profile {
  std::string canonical,play,schema,topic;
  int width,stage;
  bool heightMemory,fasterHeight;
};
inline Profile profileFor(const std::string& task){
  const std::string root="Isaac-NANDU1-Blind-Flat-";
  struct Entry{int suffix,width,stage;const char* schema;bool memory,fast;};
  for(const auto& e:std::array<Entry,5>{{
      {2,70,4,"nandu_network_v1_3_1_2_2_E_adaptive_balance_r2",false,false},
      {3,71,1,"nandu_network_v1_3_1_2_3_ablation_B_r2",false,false},
      {5,71,4,"nandu_network_v1_3_1_2_5_ablation_D_r2",true,false},
      {6,71,4,"nandu_network_v1_3_1_2_6_ablation_E_r2",true,true},
      {7,71,2,"nandu_network_v1_3_1_2_7_ablation_F_r2",true,false}}}){
    std::string suffix="v1-3-1-2-"+std::to_string(e.suffix),canonical=root+suffix,play=root+"Play-"+suffix;
    if(task==canonical||task==play)return {canonical,play,e.schema,"/nandu1/v1_3_1_2_"+std::to_string(e.suffix)+"/command",e.width,e.stage,e.memory,e.fast};
  }
  throw std::runtime_error("Unsupported adaptive deployment task: "+task);
}
}
