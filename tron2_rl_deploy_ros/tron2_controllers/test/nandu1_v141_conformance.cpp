#include "tron2_controllers/NANDU1V141.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  if(argc!=2)return 2;
  try{
    std::ifstream input(argv[1]);int count;input>>count;
    nandu1::v141::CommandProcessor p;float maximum=0;
    auto compare=[&](float actual,float expected){float error=std::abs(actual-expected);maximum=std::max(maximum,error);if(error>3e-5f+1e-5f*std::abs(expected))throw std::runtime_error("v1.4.1 conformance failed");};
    for(int i=0;i<count;++i){
      int stage,reset;float height,dt;input>>stage>>reset>>height>>dt;
      Eigen::Vector3f gravity,gyro;for(int j=0;j<3;++j)input>>gravity[j];for(int j=0;j<3;++j)input>>gyro[j];
      std::array<double,3> command;for(auto& x:command)input>>x;
      p.setStage(stage);if(reset)p.reset(height);p.setExternal(command);
      for(float x:p.step(gravity,gyro,dt)){float expected;input>>expected;compare(x,expected);}
      float heading,target;input>>heading>>target;compare(p.heading(),heading);compare(p.target(),target);
      // A state restore must yield exactly the same continuation and tick latch.
      auto snapshot=p.state();auto restored=p;restored.restore(snapshot,p.schema);
      auto first=p.step(gravity,gyro,dt,i),second=restored.step(gravity,gyro,dt,i);
      for(int j=0;j<6;++j)compare(first[j],second[j]);
      for(int j=0;j<6;++j)compare(first[j],p.step(gravity,gyro,dt,i)[j]);
      // Restore the original tested stream after the extra continuation check.
      p.restore(snapshot,p.schema);
    }
    if(!input)throw std::runtime_error("Truncated vectors");
    int adapterRows=0;
    if(input>>adapterRows){
      for(int i=0;i<adapterRows;++i){
        std::vector<float> enc(7),obs(49),cmd(6);for(auto& x:enc)input>>x;for(auto& x:obs)input>>x;for(auto& x:cmd)input>>x;
        for(float x:nandu1::v141::actorInput(enc,obs,cmd)){float expected;input>>expected;compare(x,expected);}
      }
      if(!input)throw std::runtime_error("Truncated adapter vectors");
    }
    p.setExternal({0,0,.9},true);if(p.step({0,0,-1},{0,0,0})[5]!=0)throw std::runtime_error("Step height rate");
    std::cout<<"v1.4.1 processor maximum absolute error "<<maximum<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
