#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "params.h"
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
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: rt_winding program.ptx input.bin output.bin\n"; return 2; }
    try {
        std::ifstream input(argv[2], std::ios::binary);
        uint32_t header[2]{};
        input.read(reinterpret_cast<char*>(header), sizeof(header));
        const uint32_t triangles = header[0], rays = header[1];
        if (!triangles || !rays || triangles > 10000000 || rays > 100000000)
            throw std::runtime_error("invalid fixture size");
        std::vector<float3> corners(size_t(triangles)*3), origins(rays);
        input.read(reinterpret_cast<char*>(corners.data()), corners.size()*sizeof(float3));
        input.read(reinterpret_cast<char*>(origins.data()), origins.size()*sizeof(float3));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("invalid fixture payload");
        std::ifstream ptx_file(argv[1]);
        std::string ptx{std::istreambuf_iterator<char>(ptx_file), {}};
        if (ptx.empty()) throw std::runtime_error("missing program");
        check(cudaFree(nullptr));
        check(optixInit());
        OptixDeviceContext context{};
        OptixDeviceContextOptions options{};
        check(optixDeviceContextCreate(nullptr, &options, &context));
        cudaStream_t stream{};
        check(cudaStreamCreate(&stream));
        CUdeviceptr vertices = allocate(corners.size()*sizeof(float3), corners.data());
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
        CUdeviceptr scratch = allocate(sizes.tempSizeInBytes);
        CUdeviceptr storage = allocate(sizes.outputSizeInBytes);
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
        OptixModule module{};
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
        OptixProgramGroup groups[3]{};
        OptixProgramGroupOptions group_options{};
        check(optixProgramGroupCreate(context, descriptions, 3, &group_options, nullptr, nullptr, groups));
        OptixPipelineLinkOptions link{};
        link.maxTraceDepth = 1;
        OptixPipeline pipeline{};
        check(optixPipelineCreate(context, &compile, &link, groups, 3, nullptr, nullptr, &pipeline));
        OptixStackSizes stack{};
        for (auto group : groups) check(optixUtilAccumulateStackSizes(group, &stack, pipeline));
        unsigned traversal{}, state{}, continuation{};
        check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
        check(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1));
        Record records[3]{};
        for (int i=0; i<3; ++i) check(optixSbtRecordPackHeader(groups[i], &records[i]));
        CUdeviceptr sbt_data = allocate(sizeof(records), records);
        OptixShaderBindingTable sbt{};
        sbt.raygenRecord = sbt_data;
        sbt.missRecordBase = sbt_data + sizeof(Record);
        sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = sbt_data + 2*sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = 1;
        CUdeviceptr ray_data = allocate(origins.size()*sizeof(float3), origins.data());
        CUdeviceptr output = allocate(size_t(rays)*sizeof(int));
        WindingParams params{gas, reinterpret_cast<float3*>(vertices), reinterpret_cast<float3*>(ray_data), reinterpret_cast<int*>(output)};
        CUdeviceptr params_data = allocate(sizeof(params), &params);
        check(optixLaunch(pipeline, stream, params_data, sizeof(params), &sbt, rays, 1, 1));
        check(cudaStreamSynchronize(stream));
        std::vector<int> result(rays);
        check(cudaMemcpy(result.data(), reinterpret_cast<void*>(output), result.size()*sizeof(int), cudaMemcpyDeviceToHost));
        std::ofstream out(argv[3], std::ios::binary);
        out.write(reinterpret_cast<char*>(result.data()), result.size()*sizeof(int));
        if (!out) throw std::runtime_error("output write failed");
        // Single-shot process: release explicit resources before context teardown.
        for (auto ptr : {params_data, output, ray_data, sbt_data, scratch, storage, vertices})
            check(cudaFree(reinterpret_cast<void*>(ptr)));
        check(optixPipelineDestroy(pipeline));
        for (auto group : groups) check(optixProgramGroupDestroy(group));
        check(optixModuleDestroy(module));
        check(optixDeviceContextDestroy(context));
        check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
