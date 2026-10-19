#include "tron2_controllers/NANDU1AdaptiveRuntime.h"
#include "tron2_controllers/NANDU1AdaptiveProfiles.h"
#include <fstream>
#include <iostream>
using namespace nandu1::adaptive;
static Vec read(std::istream& in,int n){Vec v(n);for(auto& x:v)if(!(in>>x))throw std::runtime_error("Truncated runtime fixture");return v;}
static float maximum=0,absoluteTolerance=1e-5f;
static void check(const Vec& actual,const Vec& expected,const std::string& field,int row){
  if(actual.size()!=expected.size())throw std::runtime_error("Wrong width "+field);
  for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-expected[i]);maximum=std::max(maximum,e);
    if(!std::isfinite(actual[i])||e>absoluteTolerance+1e-4f*std::abs(expected[i]))throw std::runtime_error("cycle="+std::to_string(row)+" field="+field+" index="+std::to_string(i)+" got="+std::to_string(actual[i])+" expected="+std::to_string(expected[i]));}
}
int main(int argc,char** argv){try{
  if(argc!=2&&argc!=3)throw std::runtime_error("BUNDLE [70|71|TASK] required");
  std::string selected=argc==3?argv[2]:"71";
  if(selected=="70")selected="Isaac-NANDU1-Blind-Flat-v1-3-1-2-2";
  if(selected=="71")selected="Isaac-NANDU1-Blind-Flat-v1-3-1-2-6";
  auto profile=profileFor(selected);int width=profile.width;
  std::string path=argv[1];Networks networks(path,width);Runtime r(width,profile.heightMemory,profile.fasterHeight);
  if(selected!="Isaac-NANDU1-Blind-Flat-v1-3-1-2-6")absoluteTolerance=1e-4f;
  std::ifstream in(path+"/runtime_conformance.txt");int rows,columns;in>>rows>>columns;
  if(rows<1000||columns!=(width==71?482:479))throw std::runtime_error("Invalid conformance header");
  for(int row=0;row<rows;++row){int reset,stage,clipped,id;in>>reset>>stage>>clipped>>id;
    auto obs=read(in,49),goal=read(in,3);if(reset)r.reset(.9f,stage);
    r.reportIntervalClipping(clipped,false);auto result=r.step(obs,goal,id,networks);
    for(const auto& field:std::vector<std::pair<std::string,Vec>>{{"commands",result.commands},{"encoded",result.encoded},
        {"dynamic",result.dynamic},{"hidden",result.hidden},{"eta",result.eta},{"actor_input",result.actorInput},
        {"action",result.action},{"history",result.history},{"memory_filter",result.memoryFilter},{"height_filter",result.heightFilter}})
      check(field.second,read(in,field.second.size()),field.first,row);
    check(r.step(obs,{1,1,1.4},id,networks).action,result.action,"repeat id",row);
  }
  std::cout<<"Adaptive runtime: "<<rows<<" cycles PASS; max absolute error="<<maximum<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
