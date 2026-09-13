#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "../rt_winding_v1/params.h"
#include "api.h"
#include <cmath>
#include <thread>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <cstdint>

static void check(OptixResult r) {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(optixGetErrorName(r));
}
static void check(cudaError_t r) {
    if (r != cudaSuccess) throw std::runtime_error(cudaGetErrorString(r));
}
static CUdeviceptr allocate(size_t bytes, const void* source = nullptr) {
    void* pointer = nullptr;
    check(cudaMalloc(&pointer, bytes));
    if (source) check(cudaMemcpy(pointer, source, bytes, cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(pointer);
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

struct State {
    OptixDeviceContext context{};
    cudaStream_t stream{};
    OptixModule module{};
    OptixProgramGroup groups[3]{};
    OptixPipeline pipeline{};
    OptixShaderBindingTable sbt{};
    CUdeviceptr vertices{},scratch{},storage{},sbt_data{},ray_data{},output{},params_data{};
    uint32_t capacity{};
    std::thread::id owner=std::this_thread::get_id();
    void initialize(const std::string& ptx,const std::vector<float3>& corners) {
        check(cudaFree(nullptr));
        check(optixInit());
        
        OptixDeviceContextOptions options{};
        check(optixDeviceContextCreate(nullptr, &options, &context));
        
        check(cudaStreamCreate(&stream));
        vertices = allocate(corners.size()*sizeof(float3), corners.data());
        const unsigned flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
        OptixBuildInput build{};
        build.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build.triangleArray.vertexBuffers = &vertices;
        build.triangleArray.numVertices = corners.size();
        build.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        build.triangleArray.flags = &flags;
        build.triangleArray.numSbtRecords = 1;
        OptixAccelBuildOptions accel{};
        accel.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        check(optixAccelComputeMemoryUsage(context, &accel, &build, 1, &sizes));
        scratch = allocate(sizes.tempSizeInBytes);
        storage = allocate(sizes.outputSizeInBytes);
        OptixTraversableHandle gas{};
        check(optixAccelBuild(context, stream, &accel, &build, 1, scratch,
              sizes.tempSizeInBytes, storage, sizes.outputSizeInBytes, &gas, nullptr, 0));
        OptixPipelineCompileOptions compile{};
        compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile.numPayloadValues = 1;
        compile.numAttributeValues = 2;
        compile.pipelineLaunchParamsVariableName = "params";
        compile.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        OptixModuleCompileOptions module_options{};
        
        check(optixModuleCreate(context, &module_options, &compile,
                               ptx.data(), ptx.size(), nullptr, nullptr, &module));
        OptixProgramGroupDesc descriptions[3]{};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen = {module, "__raygen__winding"};
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss = {module, "__miss__winding"};
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleAH = module;
        descriptions[2].hitgroup.entryFunctionNameAH = "__anyhit__winding";
        
        OptixProgramGroupOptions group_options{};
        check(optixProgramGroupCreate(context, descriptions, 3, &group_options, nullptr, nullptr, groups));
        OptixPipelineLinkOptions link{};
        link.maxTraceDepth = 1;
        
        check(optixPipelineCreate(context, &compile, &link, groups, 3, nullptr, nullptr, &pipeline));
        OptixStackSizes stack{};
        for (auto group : groups) check(optixUtilAccumulateStackSizes(group, &stack, pipeline));
        unsigned traversal{}, state{}, continuation{};
        check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
        check(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1));
        Record records[3]{};
        for (int i=0; i<3; ++i) check(optixSbtRecordPackHeader(groups[i], &records[i]));
        sbt_data = allocate(sizeof(records), records);
        
        sbt.raygenRecord = sbt_data;
        sbt.missRecordBase = sbt_data + sizeof(Record);
        sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = sbt_data + 2*sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = 1;
        ray_data = allocate(size_t(capacity)*sizeof(float3));
        output = allocate(size_t(capacity)*sizeof(int));
        WindingParams params{gas, reinterpret_cast<float3*>(vertices), reinterpret_cast<float3*>(ray_data), reinterpret_cast<int*>(output)};
        params_data = allocate(sizeof(params), &params);

    }
    int release() noexcept {
        int failed=0;
        if(stream && cudaStreamSynchronize(stream)!=cudaSuccess) failed=2;
        for(auto ptr:{params_data,output,ray_data,sbt_data,scratch,storage,vertices})
            if(ptr && cudaFree(reinterpret_cast<void*>(ptr))!=cudaSuccess) failed=2;
        if(pipeline && optixPipelineDestroy(pipeline)!=OPTIX_SUCCESS) failed=2;
        for(auto group:groups) if(group && optixProgramGroupDestroy(group)!=OPTIX_SUCCESS) failed=2;
        if(module && optixModuleDestroy(module)!=OPTIX_SUCCESS) failed=2;
        if(context && optixDeviceContextDestroy(context)!=OPTIX_SUCCESS) failed=2;
        if(stream && cudaStreamDestroy(stream)!=cudaSuccess) failed=2;
        return failed;
    }
};
extern "C" int rt_create(const char* path,const float* input,uint32_t triangles,uint32_t capacity,void** handle) {
    if(!handle || *handle || !path || !input || !triangles || triangles>10000000 || !capacity || capacity>100000000) return 1;
    for(size_t i=0;i<size_t(triangles)*9;++i) if(!std::isfinite(input[i])) return 1;
    State* state=nullptr;
    try {
        std::ifstream file(path);
        std::string ptx{std::istreambuf_iterator<char>(file),{}};
        if(ptx.empty()) return 1;
        std::vector<float3> corners(size_t(triangles)*3);
        for(size_t i=0;i<corners.size();++i) corners[i]=make_float3(input[3*i],input[3*i+1],input[3*i+2]);
        state=new State;state->capacity=capacity;state->initialize(ptx,corners);
        *handle=state; return 0;
    } catch(...) { if(state){state->release();delete state;}return 2; }
}
extern "C" int rt_query(void* handle,const float* origins,uint32_t count,int32_t* result) {
    if(!handle || !origins || !result) return 1;
    State& s=*static_cast<State*>(handle);
    if(s.owner!=std::this_thread::get_id()) return 3;
    if(!count || count>s.capacity) return 1;
    for(size_t i=0;i<size_t(count)*3;++i) if(!std::isfinite(origins[i])) return 1;
    try {
        check(cudaMemcpy(reinterpret_cast<void*>(s.ray_data),origins,size_t(count)*sizeof(float3),cudaMemcpyHostToDevice));
        check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,count,1,1));
        check(cudaStreamSynchronize(s.stream));
        check(cudaMemcpy(result,reinterpret_cast<void*>(s.output),size_t(count)*sizeof(int32_t),cudaMemcpyDeviceToHost));
        return 0;
    } catch(...) {return 2;}
}
extern "C" int rt_destroy(void** handle) {
    if(!handle) return 1;
    if(!*handle) return 0;
    State* s=static_cast<State*>(*handle);
    if(s->owner!=std::this_thread::get_id()) return 3;
    int status=s->release();delete s;*handle=nullptr;return status;
}
