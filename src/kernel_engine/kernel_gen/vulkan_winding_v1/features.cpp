// Feature inventory only. No dispatch, implicit file writes or hardware changes.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
#include <cstring>
int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName="winding-feature-inventory"; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo=&app;
    VkInstance instance{}; VkResult rc=vkCreateInstance(&ci,nullptr,&instance);
    if(rc!=VK_SUCCESS){std::printf("{\"instance_error\":%d}\n",int(rc));return 2;}
    uint32_t n=0; rc=vkEnumeratePhysicalDevices(instance,&n,nullptr);
    if(rc!=VK_SUCCESS){vkDestroyInstance(instance,nullptr);return 3;}
    std::vector<VkPhysicalDevice> devices(n);
    rc=vkEnumeratePhysicalDevices(instance,&n,devices.data());
    if(rc!=VK_SUCCESS){vkDestroyInstance(instance,nullptr);return 3;}
    std::printf("{\"devices\":["); bool usable=false;
    for(uint32_t i=0;i<n;i++){
        VkPhysicalDeviceRayQueryFeaturesKHR ray{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR}; as.pNext=&ray;
        VkPhysicalDeviceBufferDeviceAddressFeatures address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};address.pNext=&as;
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&address;
        vkGetPhysicalDeviceFeatures2(devices[i],&f);
        VkPhysicalDeviceFloatControlsProperties fp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};prop.pNext=&fp;
        vkGetPhysicalDeviceProperties2(devices[i],&prop);
        uint32_t count=0;rc=vkEnumerateDeviceExtensionProperties(devices[i],nullptr,&count,nullptr);
        if(rc!=VK_SUCCESS){vkDestroyInstance(instance,nullptr);return 4;}
        std::vector<VkExtensionProperties> ext(count);
        rc=vkEnumerateDeviceExtensionProperties(devices[i],nullptr,&count,ext.data());
        if(rc!=VK_SUCCESS){vkDestroyInstance(instance,nullptr);return 4;}
        bool ray_ext=false,as_ext=false;
        for(auto& e:ext){ray_ext|=!std::strcmp(e.extensionName,VK_KHR_RAY_QUERY_EXTENSION_NAME);as_ext|=!std::strcmp(e.extensionName,VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);}
        bool hardware=prop.properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || prop.properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        usable|=hardware && ray_ext && as_ext && ray.rayQuery && as.accelerationStructure && address.bufferDeviceAddress && f.features.shaderFloat64;
        std::printf("%s{\"type\":%u,\"api\":%u,\"ray_extension\":%u,\"as_extension\":%u,\"ray_query\":%u,\"acceleration_structure\":%u,\"buffer_device_address\":%u,\"float64\":%u,\"float32_denorm_preserve\":%u,\"float32_flush_zero\":%u}",
            i?",":"",unsigned(prop.properties.deviceType),prop.properties.apiVersion,unsigned(ray_ext),unsigned(as_ext),ray.rayQuery,as.accelerationStructure,address.bufferDeviceAddress,f.features.shaderFloat64,fp.shaderDenormPreserveFloat32,fp.shaderDenormFlushToZeroFloat32);
    }
    std::printf("],\"hardware_ready\":%s}\n",usable?"true":"false");
    vkDestroyInstance(instance,nullptr);return usable?0:2;
}
