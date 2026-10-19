#include "tron2_controllers/NANDU1Profiles.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  if(argc!=2)return 2;
  try{
    std::ifstream input(std::string(argv[1])+"/yaw_conformance.txt");int stage,count;input>>stage>>count;
    if(!input||count<=0)throw std::runtime_error("Missing yaw vectors");
    nandu1::v131::CommandProcessor p;p.configure(stage,.9);float maximum=0;
    auto compare=[&](float actual,float expected){float error=std::abs(actual-expected);maximum=std::max(maximum,error);if(error>3e-5f+1e-5f*std::abs(expected))throw std::runtime_error("TCP yaw conformance failed");};
    for(int i=0;i<count;++i){
      int reset;float dt;Eigen::Vector3f gravity,gyro;std::array<double,3> command;
      input>>reset>>dt;for(int j=0;j<3;++j)input>>gravity[j];for(int j=0;j<3;++j)input>>gyro[j];for(auto& x:command)input>>x;
      if(reset)p.reset(.9);
      for(float value:p.step(command,gravity,gyro,dt)){float expected;input>>expected;compare(value,expected);}
      float heading,target;input>>heading>>target;compare(p.heading(),heading);compare(p.target(),target);
    }
    if(!input)throw std::runtime_error("Truncated TCP yaw vectors");
    std::cout<<"TCP yaw maximum absolute error "<<maximum<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
