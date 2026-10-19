#include "tron2_controllers/NANDU1V14.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  if(argc!=2)return 2;
  try{
    std::ifstream f(std::string(argv[1])+"/runtime_conformance.txt");if(!f)throw std::runtime_error("Missing runtime vectors");
    int count,stage;float height;f>>count>>stage>>height;
    nandu1::v14::CommandProcessor processor;processor.configure(stage,height);double maximum=0;
    auto compare=[&](float actual,float expected){double error=std::abs(actual-expected);maximum=std::max(maximum,error);if(error>3e-5+1e-5*std::abs(expected))throw std::runtime_error("v1.4 runtime conformance failed");};
    for(int i=0;i<count;++i){
      float reset,resetHeight,dt;f>>reset>>resetHeight>>dt;
      Eigen::Vector3f gravity,gyro;for(int j=0;j<3;++j)f>>gravity[j];for(int j=0;j<3;++j)f>>gyro[j];
      std::array<double,3> command;for(auto& v:command)f>>v;
      if(reset)processor.reset(resetHeight);
      auto out=processor.step(command,gravity,gyro,dt);
      for(float v:out){float expected;f>>expected;compare(v,expected);}
      float h,t;f>>h>>t;compare(processor.heading(),h);compare(processor.target(),t);
    }
    int rows;f>>rows;
    for(int i=0;i<rows;++i){std::vector<float> enc(7),obs(49),cmd(6);for(auto& v:enc)f>>v;for(auto& v:obs)f>>v;for(auto& v:cmd)f>>v;
      for(float v:nandu1::v14::actorInput(enc,obs,cmd)){float expected;f>>expected;compare(v,expected);}}
    if(!f)throw std::runtime_error("Truncated runtime vectors");
    std::cout<<"v1.4 processor + heading adapter maximum error "<<maximum<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
