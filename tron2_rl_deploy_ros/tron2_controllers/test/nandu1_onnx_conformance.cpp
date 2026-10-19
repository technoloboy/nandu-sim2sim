#include "tron2_controllers/NANDU1Onnx.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  if(argc!=2)return 2;
  std::string root=argv[1];std::ifstream f(root+"/conformance.txt");
  if(!f)return 2;
  try{
    for(const auto& name:{"encoder","policy"}){
      int count,n,m;f>>count>>n>>m;
      nandu1::Network net(root+"/"+name+".onnx",n,m);
      double maximum=0;
      for(int row=0;row<count;++row){
        std::vector<float> x(n),y(m);for(auto& v:x)f>>v;for(auto& v:y)f>>v;
        auto result=net.run(x);
        for(int i=0;i<m;++i){double e=std::abs(result[i]-y[i]);maximum=std::max(maximum,e);
          if(e>1e-5+1e-4*std::abs(y[i]))throw std::runtime_error("Conformance failed");}
      }
      std::cout<<name<<" maximum error "<<maximum<<"\n";
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
