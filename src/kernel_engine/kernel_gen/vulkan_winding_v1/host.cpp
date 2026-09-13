// Separate Vulkan signed-winding executable; the OptiX baseline is unchanged.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("Vulkan result "+std::to_string(r));}
struct Buffer {VkBuffer buffer{};VkDeviceMemory memory{};void* mapped{};VkDeviceAddress address{};};
struct Runner {
    VkInstance instance{};VkPhysicalDevice physical{};VkDevice device{};VkQueue queue{};
    VkCommandPool command_pool{};VkDescriptorPool descriptor_pool{};VkDescriptorSetLayout set_layout{};
    VkPipelineLayout pipeline_layout{};VkPipeline pipeline{};VkShaderModule shader{};
    VkPhysicalDeviceMemoryProperties memory_props{};VkPhysicalDeviceProperties properties{};
    VkDeviceSize scratch_alignment=256;bool software=false;
    std::vector<Buffer> buffers;std::vector<VkAccelerationStructureKHR> structures;
    PFN_vkCreateAccelerationStructureKHR create_as{};PFN_vkDestroyAccelerationStructureKHR destroy_as{};
    PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes{};
    PFN_vkCmdBuildAccelerationStructuresKHR build_command{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR as_address{};
    ~Runner(){
        if(device){
            if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
            if(shader)vkDestroyShaderModule(device,shader,nullptr);
            if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
            if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
            if(set_layout)vkDestroyDescriptorSetLayout(device,set_layout,nullptr);
            for(auto s:structures)destroy_as(device,s,nullptr);
            for(auto b:buffers){if(b.mapped)vkUnmapMemory(device,b.memory);vkDestroyBuffer(device,b.buffer,nullptr);vkFreeMemory(device,b.memory,nullptr);}
            if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);
            vkDestroyDevice(device,nullptr);
        }
        if(instance)vkDestroyInstance(instance,nullptr);
    }
    void initialize(bool allow_software){
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;
        check(vkCreateInstance(&ci,nullptr,&instance));
        uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
        std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
        uint32_t family=0;
        for(int pass=0;pass<(allow_software?2:1) && !physical;pass++)for(auto d:devices){
            VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d,&p);
            bool hw=p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || p.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            if((pass==0 && !hw)||(pass==1 && p.deviceType!=VK_PHYSICAL_DEVICE_TYPE_CPU))continue;
            VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
            VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};as.pNext=&rq;
            VkPhysicalDeviceBufferDeviceAddressFeatures ba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};ba.pNext=&as;
            VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&ba;vkGetPhysicalDeviceFeatures2(d,&f);
            if(!rq.rayQuery || !as.accelerationStructure || !ba.bufferDeviceAddress || !f.features.shaderFloat64)continue;
            uint32_t n=0;vkGetPhysicalDeviceQueueFamilyProperties(d,&n,nullptr);
            std::vector<VkQueueFamilyProperties> qs(n);vkGetPhysicalDeviceQueueFamilyProperties(d,&n,qs.data());
            for(uint32_t j=0;j<n;j++)if(qs[j].queueFlags&VK_QUEUE_COMPUTE_BIT){physical=d;family=j;software=!hw;properties=p;break;}
            if(physical)break;
        }
        if(!physical)throw std::runtime_error("no eligible ray-query implementation");
        float priority=1;
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};q.queueFamilyIndex=family;q.queueCount=1;q.pQueuePriorities=&priority;
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};rq.rayQuery=VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};as.accelerationStructure=VK_TRUE;as.pNext=&rq;
        VkPhysicalDeviceBufferDeviceAddressFeatures ba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};ba.bufferDeviceAddress=VK_TRUE;ba.pNext=&as;
        VkPhysicalDeviceFeatures features{};features.shaderFloat64=VK_TRUE;
        const char* extensions[]={VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,VK_KHR_RAY_QUERY_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.pNext=&ba;dc.pEnabledFeatures=&features;dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&q;
        dc.enabledExtensionCount=3;dc.ppEnabledExtensionNames=extensions;check(vkCreateDevice(physical,&dc,nullptr,&device));
        vkGetDeviceQueue(device,family,0,&queue);vkGetPhysicalDeviceMemoryProperties(physical,&memory_props);
#define LOAD(member,name) member=reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device,#name));if(!member)throw std::runtime_error("missing " #name)
        LOAD(create_as,vkCreateAccelerationStructureKHR);LOAD(destroy_as,vkDestroyAccelerationStructureKHR);
        LOAD(build_sizes,vkGetAccelerationStructureBuildSizesKHR);LOAD(build_command,vkCmdBuildAccelerationStructuresKHR);
        LOAD(as_address,vkGetAccelerationStructureDeviceAddressKHR);
#undef LOAD
        VkPhysicalDeviceAccelerationStructurePropertiesKHR ap{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};pp.pNext=&ap;vkGetPhysicalDeviceProperties2(physical,&pp);
        scratch_alignment=ap.minAccelerationStructureScratchOffsetAlignment;
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=family;
        check(vkCreateCommandPool(device,&cp,nullptr,&command_pool));
    }
    Buffer allocate(VkDeviceSize size,VkBufferUsageFlags usage,const void* data=nullptr){
        Buffer b{};VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=size;ci.usage=usage|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device,&ci,nullptr,&b.buffer));
        // Register immediately so later allocation failures still release this buffer.
        buffers.push_back(b);auto& owned=buffers.back();
        VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device,b.buffer,&req);
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<memory_props.memoryTypeCount;i++)if((req.memoryTypeBits&(1u<<i)) &&
            (memory_props.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
        if(type==UINT32_MAX)throw std::runtime_error("no coherent host-visible memory type");
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        check(vkAllocateMemory(device,&ai,nullptr,&owned.memory));check(vkBindBufferMemory(device,owned.buffer,owned.memory,0));
        check(vkMapMemory(device,owned.memory,0,size,0,&owned.mapped));if(data)std::memcpy(owned.mapped,data,size);
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=owned.buffer;
        owned.address=vkGetBufferDeviceAddress(device,&info);return owned;
    }
    VkCommandBuffer begin(){
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=command_pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
        VkCommandBuffer cmd{};check(vkAllocateCommandBuffers(device,&ai,&cmd));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(cmd,&bi));return cmd;
    }
    void finish(VkCommandBuffer cmd){
        check(vkEndCommandBuffer(cmd));VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence{};check(vkCreateFence(device,&fi,nullptr,&fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;
        VkResult r=vkQueueSubmit(queue,1,&si,fence);
        if(r==VK_SUCCESS)r=vkWaitForFences(device,1,&fence,VK_TRUE,60000000000ull);
        // A timeout/device failure terminates the process; do not destroy resources still in use.
        if(r!=VK_SUCCESS){std::cerr<<"submission failed "<<r<<"\n";std::_Exit(3);}
        vkDestroyFence(device,fence,nullptr);vkFreeCommandBuffers(device,command_pool,1,&cmd);
    }
    VkAccelerationStructureKHR build(VkAccelerationStructureTypeKHR type,VkAccelerationStructureGeometryKHR geometry,uint32_t primitives){
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type=type;info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;info.geometryCount=1;info.pGeometries=&geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};build_sizes(device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&primitives,&sizes);
        Buffer storage=allocate(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
        Buffer scratch=allocate(sizes.buildScratchSize+scratch_alignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};ci.buffer=storage.buffer;ci.size=sizes.accelerationStructureSize;ci.type=type;
        VkAccelerationStructureKHR handle{};check(create_as(device,&ci,nullptr,&handle));structures.push_back(handle);
        info.dstAccelerationStructure=handle;info.scratchData.deviceAddress=(scratch.address+scratch_alignment-1)/scratch_alignment*scratch_alignment;
        VkAccelerationStructureBuildRangeInfoKHR range{};range.primitiveCount=primitives;const auto* ranges=&range;
        auto cmd=begin();build_command(cmd,1,&info,&ranges);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;barrier.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        finish(cmd);return handle;
    }
    void execute(const std::vector<float>& corners,const std::vector<float>& origins,const std::vector<uint32_t>& program,std::vector<int32_t>& counts){
        uint32_t triangles=uint32_t(corners.size()/9),nrays=uint32_t(counts.size());
        if((uint64_t(nrays)+63)/64>properties.limits.maxComputeWorkGroupCount[0])throw std::runtime_error("ray count exceeds one-dimensional dispatch limit");
        Buffer vertices=allocate(corners.size()*sizeof(float),VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,corners.data());
        VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        auto& tri=geometry.geometry.triangles;tri.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;tri.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress=vertices.address;tri.vertexStride=3*sizeof(float);tri.maxVertex=triangles*3-1;tri.indexType=VK_INDEX_TYPE_NONE_KHR;
        auto blas=build(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geometry,triangles);
        VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};address.accelerationStructure=blas;
        VkAccelerationStructureInstanceKHR inst{};inst.transform.matrix[0][0]=inst.transform.matrix[1][1]=inst.transform.matrix[2][2]=1;
        inst.mask=255;inst.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR|VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;inst.accelerationStructureReference=as_address(device,&address);
        Buffer instances=allocate(sizeof(inst),VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,&inst);
        VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};tg.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tg.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;tg.geometry.instances.data.deviceAddress=instances.address;
        auto tlas=build(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,tg,1);
        Buffer rays=allocate(origins.size()*sizeof(float),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,origins.data());
        Buffer result=allocate(counts.size()*sizeof(int32_t),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        VkDescriptorSetLayoutBinding bindings[4]{};
        for(uint32_t i=0;i<4;i++){bindings[i].binding=i;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;bindings[i].descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;}
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=4;sl.pBindings=bindings;check(vkCreateDescriptorSetLayout(device,&sl,nullptr,&set_layout));
        VkDescriptorPoolSize pool_sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=pool_sizes;check(vkCreateDescriptorPool(device,&dp,nullptr,&descriptor_pool));
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=descriptor_pool;da.descriptorSetCount=1;da.pSetLayouts=&set_layout;
        VkDescriptorSet set{};check(vkAllocateDescriptorSets(device,&da,&set));
        VkWriteDescriptorSetAccelerationStructureKHR wa{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};wa.accelerationStructureCount=1;wa.pAccelerationStructures=&tlas;
        VkDescriptorBufferInfo bi[]={{vertices.buffer,0,VK_WHOLE_SIZE},{rays.buffer,0,VK_WHOLE_SIZE},{result.buffer,0,VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[4]{};
        for(uint32_t i=0;i<4;i++){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=bindings[i].descriptorType;if(i)writes[i].pBufferInfo=&bi[i-1];else writes[i].pNext=&wa;}
        vkUpdateDescriptorSets(device,4,writes,0,nullptr);
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,4};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&set_layout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;check(vkCreatePipelineLayout(device,&pl,nullptr,&pipeline_layout));
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=program.size()*4;sm.pCode=program.data();check(vkCreateShaderModule(device,&sm,nullptr,&shader));
        VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pc.layout=pipeline_layout;pc.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pc.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pc.stage.module=shader;pc.stage.pName="main";
        check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pc,nullptr,&pipeline));
        auto cmd=begin();vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(cmd,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&nrays);vkCmdDispatch(cmd,(nrays+63)/64,1,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);finish(cmd);
        std::memcpy(counts.data(),result.mapped,counts.size()*sizeof(int32_t));
    }
};
int main(int argc,char** argv){
    if(argc!=4 && !(argc==5 && std::string(argv[4])=="--allow-software")){std::cerr<<"usage: vulkan_winding winding.spv input.bin output.bin [--allow-software]\n";return 2;}
    try{
        std::ifstream input(argv[2],std::ios::binary);uint32_t header[2]{};input.read(reinterpret_cast<char*>(header),8);
        if(!header[0]||!header[1]||header[0]>10000000||header[1]>100000000)throw std::runtime_error("invalid fixture size");
        std::vector<float> corners(size_t(header[0])*9),origins(size_t(header[1])*3);
        input.read(reinterpret_cast<char*>(corners.data()),corners.size()*4);input.read(reinterpret_cast<char*>(origins.data()),origins.size()*4);
        if(!input||input.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("invalid fixture payload");
        for(float x:corners)if(!std::isfinite(x))throw std::runtime_error("nonfinite corner");
        for(float x:origins)if(!std::isfinite(x))throw std::runtime_error("nonfinite origin");
        std::ifstream file(argv[1],std::ios::binary|std::ios::ate);auto bytes=file.tellg();
        if(bytes<=0 || bytes>64*1024*1024 || bytes%4)throw std::runtime_error("invalid shader size");
        std::vector<uint32_t> program(size_t(bytes)/4);file.seekg(0);file.read(reinterpret_cast<char*>(program.data()),bytes);if(!file)throw std::runtime_error("invalid shader payload");
        std::vector<int32_t> counts(header[1]);Runner runner;runner.initialize(argc==5);runner.execute(corners,origins,program,counts);
        std::ofstream output(argv[3],std::ios::binary);output.write(reinterpret_cast<char*>(counts.data()),counts.size()*4);output.close();if(!output)throw std::runtime_error("output write failed");
        std::cout<<"{\"software\":"<<(runner.software?"true":"false")<<",\"triangles\":"<<header[0]<<",\"rays\":"<<header[1]<<"}\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
