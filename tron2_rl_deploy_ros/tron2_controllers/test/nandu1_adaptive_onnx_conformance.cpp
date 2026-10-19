#include "tron2_controllers/NANDU1AdaptiveOnnx.h"
#include <fstream>
#include <iostream>
using namespace nandu1::adaptive;
static std::vector<float> read(std::istream& in,int n){std::vector<float> v(n);for(auto& x:v)if(!(in>>x))throw std::runtime_error("Truncated fixture");return v;}
static void check(const std::vector<float>& v,const std::vector<float>& expected){
  if(v.size()!=expected.size())throw std::runtime_error("Wrong conformance width");
  for(size_t i=0;i<v.size();++i)if(!std::isfinite(v[i])||std::abs(v[i]-expected[i])>1e-5f+1e-4f*std::abs(expected[i]))throw std::runtime_error("ONNX conformance mismatch at "+std::to_string(i));
}
int main(int argc,char** argv){try{
  if(argc!=2&&argc!=3)throw std::runtime_error("Usage: nandu1_adaptive_onnx_conformance BUNDLE [70|71]");
  const int width=argc==3?std::stoi(argv[2]):71,dynamicWidth=width-59;
  std::string path=argv[1];Networks n(path,width);std::ifstream file(path+"/onnx_conformance.txt");int rows=0;file>>rows;
  if(rows!=8&&rows!=1024)throw std::runtime_error("Wrong row count");
  Graph pipeline(path+"/actor_pipeline.onnx",{{"history",245},{"observation",49},{"commands",3},{"dynamic",dynamicWidth}},{{"actions",13}});
  for(int i=0;i<rows;++i){
    auto history=read(file,245),obs=read(file,49),command=read(file,3),dynamic=read(file,dynamicWidth),issued=read(file,13),hidden=read(file,64),eta=read(file,8);
    auto encoded=read(file,7),raw=read(file,width),action=read(file,13),nh=read(file,64),ne=read(file,8),re=read(file,8);
    check(n.encode(history),encoded);check(n.act(raw),action);auto ctx=n.context(obs,issued,hidden,eta);
    check(ctx.hidden,nh);check(ctx.filtered,ne);check(ctx.raw,re);check(pipeline.run({history,obs,command,dynamic})[0],action);
  }
  bool rejected=false;try{n.encode(std::vector<float>(244));}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("Bad width accepted");
  rejected=false;try{Graph bad(path+"/encoder.onnx",{{"WRONG",245}},{{"encoded",7}});}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("Wrong names accepted");
  auto bad=std::vector<float>(width);bad[2]=NAN;rejected=false;try{n.act(bad);}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("NaN accepted");
  std::cout<<"Adaptive ONNX: "<<rows<<" rows x4 graphs PASS; invalid names/width/NaN rejected\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
