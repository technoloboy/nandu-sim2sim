#pragma once
#include <onnxruntime_cxx_api.h>
#include "NANDU1Runtime.h"
#include <memory>
#include <string>
namespace nandu1 {
class Network {
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"nandu1"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_,output_;
  int width_,outWidth_;
public:
  Network(const std::string& path,int width,int output):width_(width),outWidth_(output){
    options_.SetIntraOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_=std::make_unique<Ort::Session>(env_,path.c_str(),options_);
    if(session_->GetInputCount()!=1 || session_->GetOutputCount()!=1)throw std::runtime_error("ONNX input/output count mismatch");
    Ort::AllocatorWithDefaultOptions allocator;
    auto in=session_->GetInputName(0,allocator);input_=in;allocator.Free(in);
    auto out=session_->GetOutputName(0,allocator);output_=out;allocator.Free(out);
    auto inputType=session_->GetInputTypeInfo(0);
    auto outputType=session_->GetOutputTypeInfo(0);
    auto i=inputType.GetTensorTypeAndShapeInfo();
    auto o=outputType.GetTensorTypeAndShapeInfo();
    if(i.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || o.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
       i.GetShape()!=std::vector<int64_t>{1,width} || o.GetShape()!=std::vector<int64_t>{1,output})
      throw std::runtime_error("NANDU1 ONNX shape/dtype mismatch");
  }
  std::vector<float> run(std::vector<float> input){
    finite(input,width_);std::array<int64_t,2> shape{1,width_};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(memory,input.data(),input.size(),shape.data(),shape.size());
    const char* ins[]={input_.c_str()};const char* outs[]={output_.c_str()};
    auto result=session_->Run(Ort::RunOptions{nullptr},ins,&tensor,1,outs,1);
    auto ptr=result.at(0).GetTensorMutableData<float>();
    std::vector<float> v(ptr,ptr+outWidth_);finite(v,outWidth_);return v;
  }
};
}
