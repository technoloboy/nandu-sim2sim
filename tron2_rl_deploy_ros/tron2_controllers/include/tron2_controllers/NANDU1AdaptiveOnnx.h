#pragma once
#include <onnxruntime_cxx_api.h>
#include "NANDU1Runtime.h"
#include <memory>
#include <string>

namespace nandu1::adaptive {
using Vec=std::vector<float>;
using Spec=std::vector<std::pair<std::string,int>>;
class Graph {
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"nandu-adaptive"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  Spec inputs_,outputs_;
public:
  Graph(const std::string& file,Spec inputs,Spec outputs):inputs_(std::move(inputs)),outputs_(std::move(outputs)){
    options_.SetIntraOpNumThreads(1);options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_=std::make_unique<Ort::Session>(env_,file.c_str(),options_);
    if(session_->GetInputCount()!=inputs_.size()||session_->GetOutputCount()!=outputs_.size())throw std::runtime_error("Adaptive ONNX input/output count mismatch");
    Ort::AllocatorWithDefaultOptions allocator;
    for(bool input:{true,false}){
      const auto& spec=input?inputs_:outputs_;
      for(size_t i=0;i<spec.size();++i){
        char* raw=input?session_->GetInputName(i,allocator):session_->GetOutputName(i,allocator);
        std::string name(raw);allocator.Free(raw);
        auto info=input?session_->GetInputTypeInfo(i):session_->GetOutputTypeInfo(i);
        auto tensor=info.GetTensorTypeAndShapeInfo();auto shape=tensor.GetShape();
        if(name!=spec[i].first||tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.size()!=2||
           (shape[0]!=1&&shape[0]!=-1)||shape[1]!=spec[i].second)throw std::runtime_error("Adaptive ONNX name/dtype/shape mismatch: "+name);
      }
    }
  }
  std::vector<Vec> run(std::vector<Vec> values){
    if(values.size()!=inputs_.size())throw std::runtime_error("Adaptive inference arity mismatch");
    std::vector<Ort::Value> tensors;std::vector<const char*> names,outnames;
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    for(size_t i=0;i<values.size();++i){
      nandu1::finite(values[i],inputs_[i].second);std::array<int64_t,2> shape{1,inputs_[i].second};
      tensors.push_back(Ort::Value::CreateTensor<float>(memory,values[i].data(),values[i].size(),shape.data(),2));
      names.push_back(inputs_[i].first.c_str());
    }
    for(const auto& spec:outputs_)outnames.push_back(spec.first.c_str());
    auto result=session_->Run(Ort::RunOptions{nullptr},names.data(),tensors.data(),tensors.size(),outnames.data(),outnames.size());
    std::vector<Vec> output;
    for(size_t i=0;i<outputs_.size();++i){
      const auto shape=result[i].GetTensorTypeAndShapeInfo().GetShape();
      if(shape!=std::vector<int64_t>{1,outputs_[i].second})throw std::runtime_error("Adaptive output batch/width mismatch");
      const float* p=result[i].GetTensorMutableData<float>();output.emplace_back(p,p+outputs_[i].second);
      nandu1::finite(output.back(),outputs_[i].second);
    }
    return output;
  }
};
struct ContextResult{Vec hidden,filtered,raw;};
class Networks {
  int width_;
  Graph encoder_,policy_,context_;
public:
  explicit Networks(const std::string& path,int width=71):width_(width),
    encoder_(path+"/encoder.onnx",{{"history",245}},{{"encoded",7}}),
    policy_(path+"/policy.onnx",{{"raw_actor_input",width}},{{"actions",13}}),
    context_(path+"/context_step.onnx",{{"observation",49},{"previous_issued_action",13},{"hidden",64},{"filtered_eta",8}},
             {{"next_hidden",64},{"next_filtered_eta",8},{"raw_eta",8}}){
    if(width!=70&&width!=71)throw std::runtime_error("Unsupported adaptive profile width");
  }
  int rawWidth()const{return width_;}
  Vec encode(const Vec& h){return encoder_.run({h})[0];}
  Vec act(const Vec& raw){return policy_.run({raw})[0];}
  ContextResult context(const Vec& obs,const Vec& action,const Vec& hidden,const Vec& eta){
    auto v=context_.run({obs,action,hidden,eta});return {std::move(v[0]),std::move(v[1]),std::move(v[2])};
  }
};
}
