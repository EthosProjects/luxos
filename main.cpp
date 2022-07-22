#define GLFW_INCLUDE_VULKAN
#define DEBUGMODE true
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <GLM/glm.hpp>
#ifdef DEBUGMODE
#include <iostream>
#endif
#include <stdexcept>         // std::exception
#include <set>               // std::vector and std::uniqueSet
#include <optional>          // std::optional
#include <limits>            // Necessary for std::numeric_limits
#include <algorithm>         // Necessary for std::clamp
#include <fstream>
#include <unordered_map>

namespace Luxos {
    class Application;
}
//This is the width and height of the window
const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;
const uint32_t MAX_FRAMES_IN_FLIGHT = 1;
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};


std::unordered_map<GLFWwindow*, Luxos::Application*> umap;
#if DEBUGMODE 
    const bool enableValidationLayers = true;
#else
    const bool enableValidationLayers = false;
#endif
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
};
//The namespace containing all 
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
namespace Luxos {
    class Application {
    private:
        GLFWwindow* window;

        vk::Instance instance;
        vk::SurfaceKHR surface;
        vk::PhysicalDevice physicalDevice;
        vk::Device device;
        vk::Queue computeQueue;
        vk::Queue presentQueue;
        
        struct Swapchain {
            vk::SwapchainKHR swapchain;
            std::vector<vk::Image> images;
            vk::Format format;
            vk::Extent2D extent;
            std::vector<VkImageView> imageViews;
            vk::ColorSpaceKHR colorSpace;
            vk::Device device = VK_NULL_HANDLE;
            struct SupportDetails {
                vk::SurfaceCapabilitiesKHR capabilities;
                std::vector<vk::SurfaceFormatKHR> formats;
                std::vector<vk::PresentModeKHR> presentModes;
            };
            static SupportDetails getSupportDetails(vk::PhysicalDevice t_physicalDevice, vk::SurfaceKHR t_surface) {
                SupportDetails details;
                details.capabilities = t_physicalDevice.getSurfaceCapabilitiesKHR(t_surface);
                details.formats = t_physicalDevice.getSurfaceFormatsKHR(t_surface);
                details.presentModes = t_physicalDevice.getSurfacePresentModesKHR(t_surface);
                return details;
            };
            Swapchain () {};
            Swapchain (vk::PhysicalDevice t_physicalDevice, vk::Device t_device, vk::SurfaceKHR t_surface, GLFWwindow* t_p_window) {
                device = t_device;
                SupportDetails supportDetails = getSupportDetails(t_physicalDevice, t_surface);
                // Pick optimal formats
                vk::SurfaceFormatKHR surfaceFormat = pickSurfaceFormat(supportDetails.formats, t_physicalDevice);
                format = surfaceFormat.format;
                colorSpace = surfaceFormat.colorSpace;
                extent = pickExtent(supportDetails.capabilities, t_p_window);
                vk::PresentModeKHR presentMode = pickPresentMode(supportDetails.presentModes);
                //Ensure that image count is the minimum count plus one to give some leeway(need to research this)
                uint32_t imageCount = supportDetails.capabilities.minImageCount + 1;
                //Ensure that the image count is not more than the max. 0 is a special value meaning unlimited
                if (supportDetails.capabilities.maxImageCount > 0 && imageCount > supportDetails.capabilities.maxImageCount) {
                    imageCount = supportDetails.capabilities.maxImageCount;
                }
                vk::SwapchainCreateInfoKHR createInfo {
                    {},
                    t_surface,                                               // Surface
                    imageCount,                                            // Image count
                    format,                                  // Image format
                    colorSpace,                              // Color space
                    extent,                                                // Image extent
                    1,                                                     // Image array layers(1 unless doing VR basically)
                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,              // Usage flags
                    vk::SharingMode::eExclusive,                           // Sharing mode
                    nullptr,                                               // Queue family indices
                    supportDetails.capabilities.currentTransform, // preTransform
                    vk::CompositeAlphaFlagBitsKHR::eOpaque,                // Alpha blending
                    presentMode,                                           // Present mode
                    VK_FALSE,                                               // Clipped
                    VK_NULL_HANDLE,                                        // Previous swapchain (has to do with resizing)
                };
                // Handle queue sharing
                QueueFamilyIndices indices = getQueueFamilyIndexes(t_physicalDevice, t_surface);
                uint32_t queueFamilyIndices[] = {indices.computeFamily.value(), indices.presentFamily.value()};
                if (indices.computeFamily != indices.presentFamily) {
                    createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
                    createInfo.queueFamilyIndexCount = 2;
                    createInfo.pQueueFamilyIndices = queueFamilyIndices;
                }
                // Actually create swapchain
                swapchain = device.createSwapchainKHR(createInfo);
                images = device.getSwapchainImagesKHR(swapchain);
                int i = 0;
                for (vk::Image image : images) {
                    imageViews.push_back(device.createImageView({
                        {},
                        image,
                        vk::ImageViewType::e2D,
                        format,
                        vk::ComponentSwizzle {},
                        vk::ImageSubresourceRange {
                            vk::ImageAspectFlagBits::eColor,
                            0,
                            1,
                            0,
                            1
                        }
                    }));
                    i++;
                };
                #if DEBUGMODE
                std::cout << "There are " << imageCount << " images in the swapchain\n";
                #endif
            };
            void destroy () {
                if (destroyed || device == NULL) return;
                else destroyed = true;
                for (auto imageView : imageViews) {
                    device.destroyImageView(imageView);
                }
                device.destroySwapchainKHR(swapchain);
                #if DEBUGMODE
                std::cout << "Destroyed swapchain objects\n";
                #endif
            };
            static vk::SurfaceFormatKHR pickSurfaceFormat(std::vector<vk::SurfaceFormatKHR> &t_availableFormats, vk::PhysicalDevice t_physicalDevice) {
                for (const auto& availableFormat : t_availableFormats) {
                    VkFormatProperties formatProperties;
                    vkGetPhysicalDeviceFormatProperties(t_physicalDevice, VK_FORMAT_R8_UNORM, &formatProperties);
                    if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                        return availableFormat;
                    }
                }
                return t_availableFormats[0];

            };
            static vk::PresentModeKHR pickPresentMode(std::vector<vk::PresentModeKHR> t_availablePresentModes) {
                for (const auto& availablePresentMode : t_availablePresentModes) {
                        std::cout << (int) availablePresentMode << "\n";
                    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                        return availablePresentMode;
                    }
                }
                return vk::PresentModeKHR::eFifo;
            };
            static vk::Extent2D pickExtent(vk::SurfaceCapabilitiesKHR t_availableCapabilities, GLFWwindow* t_p_window) {
                if (t_availableCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                    return t_availableCapabilities.currentExtent;
                } else {
                    int width, height;
                    glfwGetFramebufferSize(t_p_window, &width, &height);
                    vk::Extent2D o_actualExtent = { (uint32_t) width, (uint32_t) height };
                    o_actualExtent.width = std::clamp(o_actualExtent.width, t_availableCapabilities.minImageExtent.width, t_availableCapabilities.maxImageExtent.width);
                    o_actualExtent.height = std::clamp(o_actualExtent.height, t_availableCapabilities.minImageExtent.height, t_availableCapabilities.maxImageExtent.height);
                    return o_actualExtent;
                }
            };
        private:
            bool destroyed = false;
        };
        Swapchain swapchain;
        vk::DescriptorPool descriptorPool;
        vk::ShaderModule shaderModule;
        vk::Pipeline computePipeline;
        vk::PipelineLayout pipelineLayout;
        vk::PipelineCache pipelineCache;
        vk::DescriptorSetLayout descriptorSetLayout;

        std::vector<vk::Image> storageImages;
        std::vector<vk::DeviceMemory> storageImageMemory;
        std::vector<vk::ImageView> storageImageViews;
        std::vector<vk::DescriptorSet> descriptorSets;
        std::vector<vk::Buffer> uniformBuffers;
        std::vector<vk::DeviceMemory> uniformBuffersMemory;
        std::vector<vk::CommandBuffer> commandBuffers;
        std::vector<vk::Semaphore> imageAvailableSemaphores;
        std::vector<vk::Semaphore> renderFinishedSemaphores;
        std::vector<vk::Fence> inFlightFences;
        std::vector<vk::Fence> baseCommandsFinishedFences;
        struct VulkanFrame {

        };
        std::vector<VulkanFrame> vulkanFrames;
        vk::CommandPool commandPool;
        struct Camera {
            alignas(16) glm::vec3 positionVector;
            alignas(16) glm::vec3 lookAtVector;
            alignas(16) glm::vec3 upVector;
            float len;
            float width;
            float aspectRatio;
            
            alignas(16) glm::vec3 alignmentVector;
            alignas(16) glm::vec3 UVector;
            alignas(16) glm::vec3 VVector;
            alignas(16) glm::vec3 projectionScreenCenterVector;
            void updateGeometry() {  
                alignmentVector = glm::normalize(lookAtVector - positionVector);
                UVector = glm::normalize(glm::cross(alignmentVector, upVector)) * width;
                VVector = glm::normalize(glm::cross(UVector, alignmentVector)) * (width / aspectRatio);
                projectionScreenCenterVector = positionVector + (alignmentVector * len);
            };
        };
        Camera camera;
        // Window functions
        inline GLFWwindow* createWindow () {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        };
    public:
        void keyback(int key, int scancode, int action, int mods) {
            // if (key == GLFW_KEY_W && action == GLFW_PRESS) {
            //     glm::vec3 pointingVector = glm::normalize(camera.lookAtVector - camera.positionVector) * 0.5f;
            //     camera.positionVector += pointingVector;
            //     camera.lookAtVector += camera.lookAtVector;
            //     camera.updateGeometry();
            // };
            // if (key == GLFW_KEY_S && action == GLFW_PRESS) {
            //     glm::vec3 pointingVector = glm::normalize(camera.lookAtVector - camera.positionVector) * 0.5f;
            //     camera.positionVector -= pointingVector;
            //     camera.lookAtVector -= camera.lookAtVector;
            //     camera.updateGeometry();
            // };
            // if (key == GLFW_KEY_A && action == GLFW_PRESS) {
            //     std::cout << glm::dot(camera.lookAtVector, camera.upVector);
            //     camera.lookAtVector += glm::normalize(glm::cross(camera.alignmentVector, camera.upVector)) * 0.1f;
            //     camera.updateGeometry();
            // }
        };
    private:
        inline void initializeGLFW () {
            glfwInit();
            window = createWindow();
            if (window == nullptr) throw std::runtime_error("Failed to create window");
            glfwSetWindowPos(window, 0, 30);
            glfwSetKeyCallback(window, key_callback);
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            glfwSetCursorPos(window, width/2, height/2);
            umap[window] = this;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        };
        // Vulkan functions
        inline vk::Instance createInstance() {
            vk::ApplicationInfo applicationInfo {
                "Luxos",                  // Application name
                VK_MAKE_VERSION(0, 0, 0), // Application version
                "Pathos",                 // Engine name
                VK_MAKE_VERSION(0, 0, 0), // Engine version
                VK_API_VERSION_1_3        // Vulkan API version
            };
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            #if DEBUGMODE
            std::cout << "Required extensions:\n";
            for (int i = 0; i < glfwExtensionCount; i++) {
                std::cout << '\t' << glfwExtensions[i] << '\n';
            }
            auto extensions = vk::enumerateInstanceExtensionProperties();
            std::cout << "Available extensions:\n";
            for (const auto& extension : extensions) {
                std::cout << '\t' << extension.extensionName << " " << extension.specVersion << '\n';
            }
            #endif
            return vk::createInstance({
                {},                                 // Flags
                &applicationInfo,                   // Application Info
                (uint32_t) validationLayers.size(), // Validation layer count
                validationLayers.data(),            // Validaiton layers
                glfwExtensionCount,                 // Extension count
                glfwExtensions,                     // Extension names
                nullptr                             // pNext(idk what this is ngl)
            });
        };
        inline vk::SurfaceKHR createSurface() {
            auto p_surface = (VkSurfaceKHR) surface; // you can cast vk::* object to Vk* using a type cast
            if (glfwCreateWindowSurface(VkInstance(instance), window, nullptr, &p_surface) != VK_SUCCESS) throw std::runtime_error("Failed to create window surface!");
            return p_surface;
        };
        struct QueueFamilyIndices {
            std::optional<uint32_t> computeFamily;
            std::optional<uint32_t> presentFamily;
            inline bool isComplete() {
                return computeFamily.has_value() && presentFamily.has_value();
            }
        };
        static QueueFamilyIndices getQueueFamilyIndexes(vk::PhysicalDevice t_physicalDevice, vk::SurfaceKHR t_surface) {
            QueueFamilyIndices indices;
            auto queueFamilies = t_physicalDevice.getQueueFamilyProperties();
            int i = 0;
            for(const auto& queueFamily : queueFamilies) {
                if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) indices.computeFamily = i;
                if (t_physicalDevice.getSurfaceSupportKHR(i, t_surface)) indices.presentFamily = i;
                if (indices.isComplete()) break;
                i++;
            }
            return indices;
        };
        bool isDeviceSuitable(vk::PhysicalDevice t_physicalDevice) {
            if (!(getQueueFamilyIndexes(t_physicalDevice, surface).isComplete())) return false;
            return true;
        };
        inline vk::PhysicalDevice choosePhysicalDevice () {
            auto availablePhysicalDevices = instance.enumeratePhysicalDevices();
            if (availablePhysicalDevices.size() == 0) throw std::runtime_error("failed to find GPUs with Vulkan support!");
            #if DEBUGMODE
            std::cout << "Available devices:\n";
            #endif
            vk::PhysicalDevice o_physicalDevice;
            for (const vk::PhysicalDevice availablePhsyicalDevice: availablePhysicalDevices) {
                #if DEBUGMODE
                vk::PhysicalDeviceProperties DeviceProps = availablePhsyicalDevice.getProperties();
                const uint32_t ApiVersion = DeviceProps.apiVersion;
                vk::PhysicalDeviceLimits DeviceLimits = DeviceProps.limits;
                std::cout << "\tDevice Name    : " << DeviceProps.deviceName << "\n";
                std::cout << "\tVulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << "\n";
                std::cout << "\tMax Compute Shared Memory Size: " << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << "\n";
                std::cout << "\n";
                #endif
                // TODO: Implement device picking that gets the most preferred GPU
                if (!isDeviceSuitable(availablePhsyicalDevice)) continue;
                o_physicalDevice = availablePhsyicalDevice;
            };
            #if DEBUGMODE
            std::cout << "Chosen device: " << o_physicalDevice.getProperties().deviceName << "\n";
            #endif
            if (o_physicalDevice == NULL) throw std::runtime_error("failed to find a suitable GPU!");
            return o_physicalDevice;
        };
        inline vk::Device createLogicalDevice() {
            // Prepare queue families
            QueueFamilyIndices indices = getQueueFamilyIndexes(physicalDevice, surface);
            std::set<uint32_t> uniqueQueueFamilies = {indices.computeFamily.value(), indices.presentFamily.value()};
            std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos { };
            const float QueuePriority = 1.0f;
            for (uint32_t queueFamilyIndex : uniqueQueueFamilies) {
                queueCreateInfos.push_back({ {}, queueFamilyIndex, 1, &QueuePriority });
            };
            return physicalDevice.createDevice({
                {},                                 // Flags
                (uint32_t) queueCreateInfos.size(), // Queue count
                queueCreateInfos.data(),            // Queues create info
                (uint32_t) validationLayers.size(), // Validation layer count
                validationLayers.data(),            // Validaiton layers
                (uint32_t) deviceExtensions.size(), // Extension count
                deviceExtensions.data()             // Extensions
            });
        };
        inline vk::Queue createQueue(uint32_t t_queueFamilyIndex) {
            return device.getQueue(t_queueFamilyIndex, 0);
        };
        inline Swapchain createSwapchain() {
            return { 
                physicalDevice,
                device,
                surface,
                window
            };
        };
        inline std::vector<vk::Image> createStorageImages() {
            std::vector<vk::Image> o_storageImages;
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                o_storageImages.push_back(device.createImage({
                    {},
                    vk::ImageType::e2D,
                    vk::Format::eB8G8R8A8Unorm,
                    { WIDTH, HEIGHT, 1 },
                    1,
                    1,
                    vk::SampleCountFlagBits::e1,
                    vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                    vk::SharingMode::eExclusive,
                }));
            };
            return o_storageImages;
        };
        inline std::vector<vk::ImageView> createStorageImageViews() {
            std::vector<vk::ImageView> o_storageImageViews;
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                o_storageImageViews.push_back(device.createImageView({
                    {},
                    storageImages[i],
                    vk::ImageViewType::e2D,
                    vk::Format::eB8G8R8A8Unorm,
                    vk::ComponentSwizzle {},
                    vk::ImageSubresourceRange {
                        vk::ImageAspectFlagBits::eColor,
                        0,
                        1,
                        0,
                        1
                    }
                }));
            };
            return o_storageImageViews;
        };
        inline void allocateStorageImages() {
            storageImageMemory.resize(MAX_FRAMES_IN_FLIGHT);
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(storageImages[i]);
                vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

                uint32_t memoryTypeIndex = uint32_t(~0);
                vk::DeviceSize memoryHeapSize = uint32_t(~0);
                for (uint32_t currentMemoryTypeIndex = 0; currentMemoryTypeIndex < memoryProperties.memoryTypeCount; ++currentMemoryTypeIndex) {
                    vk::MemoryType MemoryType = memoryProperties.memoryTypes[currentMemoryTypeIndex];
                    if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
                        (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
                    {
                        memoryHeapSize = memoryProperties.memoryHeaps[MemoryType.heapIndex].size;
                        memoryTypeIndex = currentMemoryTypeIndex;
                        break;
                    }
                }
                storageImageMemory[i] = device.allocateMemory({
                    memoryRequirements.size, // Size of image
                    memoryTypeIndex          // Memory type of iamge
                });
                device.bindImageMemory(storageImages[i], storageImageMemory[i], 0);
                // DEBUG: list out stats
                #if DEBUGMODE
                std::cout << "Memory Type Index    : " << memoryTypeIndex << "\n";
                std::cout << "Memory Heap Size     : " << memoryHeapSize / 1024.f / 1024.f / 1024.f << " GB\n";
                std::cout << "Required Memory Size : " << memoryRequirements.size / 1024.f / 1024.f << " MB \n";
                #endif

            }
        };
        inline vk::DescriptorSetLayout createDescriptorSetLayout() {
            // Create descriptor set layout
            std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings {
                {
                    0,
                    vk::DescriptorType::eStorageImage, 
                    1, 
                    vk::ShaderStageFlagBits::eCompute
                },
                vk::DescriptorSetLayoutBinding {
                    1,
                    vk::DescriptorType::eUniformBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
                }
            };
            vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
                {},
                descriptorSetLayoutBindings
            );
            return device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
        };
        inline vk::ShaderModule createShaderModule() {
            auto computeShaderCode = readFile("Square.spv");
            return device.createShaderModule({
                {},
                computeShaderCode.size(),
                reinterpret_cast<const uint32_t*>(computeShaderCode.data())
            });
        };
        inline vk::PipelineLayout createPipelineLayout() {
            return device.createPipelineLayout({ {}, descriptorSetLayout });
        };
        inline vk::PipelineCache createPipelineCache() {
            return device.createPipelineCache({});
        };
        inline vk::Pipeline createComputePipeline() {
            vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo {
                {},
                vk::ShaderStageFlagBits::eCompute,
                shaderModule, 
                "main"
            };
            vk::ComputePipelineCreateInfo computePipelineCreateInfo {
                {},
                pipelineShaderStageCreateInfo,
                pipelineLayout
            };
            return device.createComputePipeline(pipelineCache, computePipelineCreateInfo).value;
        };
        inline vk::DescriptorPool createDescriptorPool() {
            // Create descriptor pool
            vk::DescriptorPoolSize descriptorPoolSize { vk::DescriptorType::eStorageImage, MAX_FRAMES_IN_FLIGHT };
            vk::DescriptorPoolSize cameraDescriptorPoolSize { vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT };
            vk::DescriptorPoolSize descriptorPoolSizes[] {descriptorPoolSize, cameraDescriptorPoolSize};
            vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo {
                {}, 
                MAX_FRAMES_IN_FLIGHT,
                2,
                descriptorPoolSizes
            };
            return device.createDescriptorPool(descriptorPoolCreateInfo);
        };
        inline std::vector<vk::DescriptorSet> createDescriptorSet() {
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
            vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo { descriptorPool, MAX_FRAMES_IN_FLIGHT, layouts.data() };
            const std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
            return descriptorSets;
        };
        inline void makeDescriptorsWritable() {
            std::vector<vk::DescriptorImageInfo> imageInfos;
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vk::DescriptorImageInfo imageInfo {
                    VK_NULL_HANDLE,
                    storageImageViews[i],
                    vk::ImageLayout::eGeneral
                };
                vk::DescriptorBufferInfo bufferInfo {
                    uniformBuffers[i],
                    0,
                    sizeof(Camera)
                };
                vk::WriteDescriptorSet descriptorWrite{};
                descriptorWrite.dstSet = descriptorSets[i];
                descriptorWrite.dstBinding = 1;
                descriptorWrite.dstArrayElement = 0;
                descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
                descriptorWrite.descriptorCount = 1;
                descriptorWrite.pBufferInfo = &bufferInfo;
                descriptorWrite.pImageInfo = nullptr; // Optional
                descriptorWrite.pTexelBufferView = nullptr; // Optional
                device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
                device.updateDescriptorSets({ 
                    vk::WriteDescriptorSet {
                        descriptorSets[i],
                        0,
                        (uint32_t) 0,
                        (uint32_t) 1,
                        vk::DescriptorType::eStorageImage,
                        &imageInfo
                    }
                }, {});
            }
        };
        inline vk::CommandPool createCommandPool() {
            // TODO: return command pool
            QueueFamilyIndices queueFamilyIndices = getQueueFamilyIndexes(physicalDevice, surface);
            vk::CommandPoolCreateInfo poolInfo {
                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                queueFamilyIndices.computeFamily.value()
            };
            vk::CommandPool o_commandPool;
            if (device.createCommandPool(&poolInfo, nullptr, &o_commandPool) != vk::Result::eSuccess) throw std::runtime_error("failed to create command pool");
            return o_commandPool;
        };
        inline std::vector<vk::CommandBuffer> createCommandBuffer() {
            vk::CommandBufferAllocateInfo commandBufferAllocateInfo {
                commandPool,
                vk::CommandBufferLevel::ePrimary,
                MAX_FRAMES_IN_FLIGHT
            };
            const std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
            return commandBuffers;
        };
        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
            vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }
            throw std::runtime_error("failed to find suitable memory type!");
        }
        void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
            vk::BufferCreateInfo bufferInfo{
                {},
                size,
                usage,
                vk::SharingMode::eExclusive
            };
            if (device.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess) throw std::runtime_error("failed to create buffer!");
            vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);
            vk::MemoryAllocateInfo allocateInfo {
                memoryRequirements.size,
                findMemoryType(memoryRequirements.memoryTypeBits, properties)
            };
            if (device.allocateMemory(&allocateInfo, nullptr, &bufferMemory) != vk::Result::eSuccess) throw std::runtime_error("failed to allocate buffer memory!");
            device.bindBufferMemory(buffer, bufferMemory, 0);
        }
        inline void createUniformBuffers() {
            VkDeviceSize bufferSize = sizeof(Camera);

            uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
            }
        };
        inline void createSyncObjects() {
            imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
            baseCommandsFinishedFences.resize(MAX_FRAMES_IN_FLIGHT);
            vk::SemaphoreCreateInfo semaphoreInfo { };
            vk::FenceCreateInfo fenceInfo {
                vk::FenceCreateFlagBits::eSignaled
            };
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
                renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                inFlightFences[i] = device.createFence(fenceInfo);
                baseCommandsFinishedFences[i] = device.createFence({});
                // TODO: Implement error handling
                /*if (imageAvailableSemaphores[i] == VK_NULL_HANDLE || 
                    renderFinishedSemaphores[i] == VK_NULL_HANDLE || 
                    inFlightFences[i] == VK_NULL_HANDLE) {
                    throw std::runtime_error("failed to create synchronization objects for a frame!");
                }*/
            }
        };
        uint32_t currentFrame = 0;
        void generateBaseCommands() {
            // Change image format 
            
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vk::ImageMemoryBarrier undefinedToGeneralBarrier(
                    {}, 
                    {}, 
                    vk::ImageLayout::eUndefined, 
                    vk::ImageLayout::eGeneral,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, storageImages[i],
                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
                vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
                commandBuffers[i].begin(CmdBufferBeginInfo);
                commandBuffers[i].pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlagBits::eByRegion,
                    0,
                    nullptr,
                    0,
                    nullptr,
                    1,
                    &undefinedToGeneralBarrier
                );
                commandBuffers[i].end();
                vk::SubmitInfo submitInfo { 
                    0,			// Num Wait Semaphores
                    nullptr,		// Wait Semaphores
                    nullptr,		// Pipeline Stage Flags
                    1,			// Num Command Buffers
                    &commandBuffers[i] };  // List of command buffers
                
                computeQueue.submit({ submitInfo }, baseCommandsFinishedFences[i]);

            }
        }
        inline void initializeVulkan () {
            instance = createInstance();
            surface = createSurface();
            physicalDevice = choosePhysicalDevice();
            device = createLogicalDevice();
            QueueFamilyIndices indices = getQueueFamilyIndexes(physicalDevice, surface);
            computeQueue = createQueue(indices.computeFamily.value());
            presentQueue = createQueue(indices.presentFamily.value());
            swapchain = createSwapchain();
            storageImages = createStorageImages();
            allocateStorageImages();
            storageImageViews = createStorageImageViews();
            descriptorSetLayout = createDescriptorSetLayout();
            shaderModule = createShaderModule();
            pipelineLayout = createPipelineLayout();
            pipelineCache = createPipelineCache();
            computePipeline = createComputePipeline();
            commandPool = createCommandPool();
            commandBuffers = createCommandBuffer();
            descriptorPool = createDescriptorPool();
            descriptorSets = createDescriptorSet();
            createCamera();
            createUniformBuffers();
            makeDescriptorsWritable();
            createSyncObjects();
            generateBaseCommands();
            if (device.waitForFences(MAX_FRAMES_IN_FLIGHT, baseCommandsFinishedFences.data(), VK_TRUE, UINT64_MAX) == vk::Result::eErrorDeviceLost) throw std::runtime_error("device lost!");
        };
        void recreateSwapChain() {
            #if DEBUGMODE
            std::cout << "window resized\n";
            #endif
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }

            vkDeviceWaitIdle(device);
            swapchain.destroy();
            swapchain = createSwapchain();
        }
        void updateUniformBuffer(uint32_t currentImage) {
            void* data;
            device.mapMemory(uniformBuffersMemory[currentImage], 0, sizeof(camera), vk::MemoryMapFlags(), &data);
            memcpy(data, &camera, sizeof(camera));
            device.unmapMemory(uniformBuffersMemory[currentImage]);
        }
        inline void drawFrame() {
            device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
            device.resetFences(1, &inFlightFences[currentFrame]);
            vk::ResultValue swapchainImageResult = device.acquireNextImageKHR(swapchain.swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE);
            if (swapchainImageResult.result == vk::Result::eErrorOutOfDateKHR) {
                recreateSwapChain();
                return;
            } else if (swapchainImageResult.result != vk::Result::eSuccess && swapchainImageResult.result != vk::Result::eSuboptimalKHR) {
                throw std::runtime_error("failed to acquire swap chain image!");
            }
            uint32_t imageIndex = swapchainImageResult.value;
            vk::Image swapchainImage = swapchain.images[imageIndex];
            // Do layout transition 
            vk::ImageMemoryBarrier undefinedToOptimizedDestinationBarrier(
                {}, 
                {}, 
                vk::ImageLayout::eUndefined, 
                vk::ImageLayout::eTransferDstOptimal,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapchainImage,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            vk::ImageMemoryBarrier optimizedDestinationToPresentBarrier(
                {}, 
                {}, 
                vk::ImageLayout::eTransferDstOptimal, 
                vk::ImageLayout::ePresentSrcKHR,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapchainImage,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            //Record command buffer
            commandBuffers[currentFrame].reset();
            vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffers[currentFrame].begin(CmdBufferBeginInfo);
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
            commandBuffers[currentFrame].bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,	// Bind point
                pipelineLayout,				        // Pipeline Layout
                0,								    // First descriptor set
                { descriptorSets[currentFrame] },	// List of descriptor sets
                {});								// Dynamic offsets
            commandBuffers[currentFrame].dispatch(WIDTH/16, (HEIGHT + HEIGHT % 16)/16, 1);
            //TODO: add support for when window is greater sized than the storage image
            vk::ImageCopy imageCopyRegion {
                {   
                    vk::ImageAspectFlagBits::eColor,
                    0, 
                    0,
                    1
                }, 
                {
                    
                }, 
                {            
                    vk::ImageAspectFlagBits::eColor,
                    0, 
                    0,
                    1
                },
                { 0, 0, 0 },
                { swapchain.extent.width, swapchain.extent.height, 1 }
            };
            commandBuffers[currentFrame].pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlagBits::eByRegion,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &undefinedToOptimizedDestinationBarrier
            );
            commandBuffers[currentFrame].copyImage(storageImages[currentFrame], vk::ImageLayout::eGeneral, swapchainImage, vk::ImageLayout::eTransferDstOptimal, 1, &imageCopyRegion);
            commandBuffers[currentFrame].pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits::eByRegion,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &optimizedDestinationToPresentBarrier
            );
            commandBuffers[currentFrame].end();
            updateUniformBuffer(currentFrame);
            vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
            vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
            vk::PipelineStageFlags waitStageFlags[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
            vk::SubmitInfo submitInfo { 
                1,			                   // Num Wait Semaphores
                waitSemaphores,		           // Wait Semaphores
                waitStageFlags,		           // Pipeline Stage Flags
                1,			                   // Num Command Buffers
                &commandBuffers[currentFrame], // List of command buffers
                1,
                signalSemaphores
            };
            
            computeQueue.submit({ submitInfo }, inFlightFences[currentFrame]);
            
            vk::PresentInfoKHR presentInfo{};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            vk::SwapchainKHR swapchains[] = {swapchain.swapchain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains;
            presentInfo.pImageIndices = &imageIndex;
            vk::Result presentResult = presentQueue.presentKHR(&presentInfo);
            if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR) {
                recreateSwapChain();
            } else if (presentResult != vk::Result::eSuccess) {
                throw std::runtime_error((int) presentResult + "failed to present swap chain image!");
            }
            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        };
        inline void update() {
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            // IDEA: create a number representign how far cursor is from the origin and then base movement on that
            float yDiff = (y * 2 /height - 1) * 10;
            float xDiff = (x * 2 /width - 1) * 10;
            
            if (x != width/2) {
                camera.lookAtVector += glm::normalize(camera.UVector) * xDiff;
                camera.updateGeometry();
            };
            if (y != height/2) {
                camera.lookAtVector += glm::normalize(camera.VVector) * yDiff;
                camera.updateGeometry();
                camera.upVector = camera.VVector;
                camera.updateGeometry();
            };
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                glm::vec3 pointingVector = camera.alignmentVector * 0.01f;
                camera.positionVector += pointingVector;
                camera.lookAtVector += pointingVector;
                camera.updateGeometry();
            };
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                glm::vec3 pointingVector = camera.alignmentVector * 0.01f;
                camera.positionVector -= pointingVector;
                camera.lookAtVector -= pointingVector;
                camera.updateGeometry();
            };
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                camera.lookAtVector -= camera.UVector * 0.01f;
                camera.positionVector -= camera.UVector * 0.01f;
                camera.updateGeometry();
            };
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                camera.lookAtVector += camera.UVector * 0.01f;
                camera.positionVector += camera.UVector * 0.01f;
                camera.updateGeometry();
            };
            glfwSetCursorPos(window, width/2, height/2);
        }
        void loop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
                update();
            }
        };
        void createCamera() {
            camera.positionVector = glm::vec3(0.0, 0.0, -10.0);
            camera.lookAtVector = glm::vec3(0.0, 0.0, 0.0);
            camera.upVector = glm::vec3(0.0, 1.0, 0.0);
            camera.len = 1.0;
            camera.width = 1.0;
            camera.aspectRatio = (16.0 / 9.0);
            camera.updateGeometry();
        };
    public:
        Application() {
            #if DEBUGMODE
            std::cout << "Creating application\n";
            #endif
        };
        void cleanup() {
            device.waitIdle();
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroyBuffer(device, uniformBuffers[i], nullptr);
                vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            }

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
                vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
                vkDestroyFence(device, inFlightFences[i], nullptr);
                vkDestroyFence(device, baseCommandsFinishedFences[i], nullptr);
            }

            vkDestroyCommandPool(device, commandPool, nullptr);
            // TODO: contain storage image in wrapper
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                device.destroyImage(storageImages[i]);
                device.freeMemory(storageImageMemory[i]);
                device.destroyImageView(storageImageViews[i]);
            }
            // TODO: contain descriptor set in wrapper
            device.destroyShaderModule(shaderModule);
            device.destroyDescriptorSetLayout(descriptorSetLayout);
            device.destroyDescriptorPool(descriptorPool);
            device.destroyPipeline(computePipeline);
            device.destroyPipelineLayout(pipelineLayout);
            device.destroyPipelineCache(pipelineCache);
            swapchain.destroy();
            device.destroy();
            instance.destroySurfaceKHR(surface);
            instance.destroy();
            glfwDestroyWindow(window);
            glfwTerminate();
        }
        void run() {
            initializeGLFW();
            initializeVulkan();
            loop();
            cleanup();
            #if DEBUGMODE
            std::cout << "Closed window";
            #endif
        };
    };
};

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Luxos::Application* app = umap[window];
    app->keyback(key, scancode, action, mods);
};
int main() {
    Luxos::Application app;
    try {
        //oldApp.run();
        app.run();
    } catch (const std::exception& e) {
        #if DEBUGMODE
        std::cerr << "\u001b[31m" << e.what() << "\u001b[0m\n";
        #endif
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}