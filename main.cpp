#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#ifdef DEBUGMODE
#include <iostream>
#endif
#include <stdexcept>         // std::exception
#include <set>               // std::vector and std::uniqueSet
#include <optional>          // std::optional
#include <limits>            // Necessary for std::numeric_limits
#include <algorithm>         // Necessary for std::clamp
#include <fstream>


//This is the width and height of the window
const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
//#define DEBUGMODE true

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
namespace Luxos {
    class Application {
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
                    if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
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
        vk::Image storageImage;
        vk::DeviceMemory storageImageMemory;
        vk::ImageView storageImageView;
        vk::Pipeline computePipeline;
        vk::PipelineLayout pipelineLayout;
        vk::DescriptorSetLayout descriptorSetLayout;
        vk::DescriptorPool descriptorPool;
        vk::DescriptorSet descriptorSet;
        vk::CommandPool commandPool;
        vk::CommandBuffer commandBuffer;
        vk::Semaphore imageAvailableSemaphore;
        vk::Semaphore renderFinishedSemaphore;
        vk::Fence inFlightFence;
        vk::Fence baseCommandsFinishedFence;


        // Window functions
        inline GLFWwindow* createWindow () {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        };
        inline void initializeGLFW () {
            glfwInit();
            window = createWindow();
            if (window == nullptr) throw std::runtime_error("Failed to create window");
            glfwSetWindowPos(window, 0, 30);
            //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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
        inline vk::Image createStorageImage() {
            return device.createImage({
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
            });
        };
        inline vk::ImageView createStorageImageView() {
            return device.createImageView({
                {},
                storageImage,
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
            });
        };
        inline void allocateStorageImage() {
            vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(storageImage);
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
            storageImageMemory = device.allocateMemory({
                memoryRequirements.size, // Size of image
                memoryTypeIndex          // Memory type of iamge
            });
            device.bindImageMemory(storageImage, storageImageMemory, 0);
            // DEBUG: list out stats
            #if DEBUGMODE
            std::cout << "Memory Type Index    : " << memoryTypeIndex << "\n";
            std::cout << "Memory Heap Size     : " << memoryHeapSize / 1024.f / 1024.f / 1024.f << " GB\n";
            std::cout << "Required Memory Size : " << memoryRequirements.size / 1024.f / 1024.f << " MB \n";
            #endif
        };
        inline void createComputePipeline() {
            // TODO: Split this function into smaller functions for more modularity and abstraction
            auto computeShaderCode = readFile("Square.spv");
            vk::ShaderModuleCreateInfo shaderModuleCreateInfo {
                {},
                computeShaderCode.size(),
                reinterpret_cast<const uint32_t*>(computeShaderCode.data())
            };
            vk::ShaderModule shaderModule = device.createShaderModule(shaderModuleCreateInfo);
            // Create descriptor set layout
            std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings {
                {
                    0,
                    vk::DescriptorType::eStorageImage, 
                    1, 
                    vk::ShaderStageFlagBits::eCompute
                }
            };
            vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
                {},
                descriptorSetLayoutBindings
            );
            descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
            // Create pipeline
            vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = { {}, descriptorSetLayout };
            pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
            vk::PipelineCache pipelineCache = device.createPipelineCache({});
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
            computePipeline = device.createComputePipeline(pipelineCache, computePipelineCreateInfo).value;
            device.destroyShaderModule(shaderModule);
            device.destroyPipelineCache(pipelineCache);
            // Create descriptor pool
            vk::DescriptorPoolSize descriptorPoolSize { vk::DescriptorType::eStorageImage, 2 };
            vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo {
                {}, 
                1,
                descriptorPoolSize
            };
            descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);
            // Allocate descriptor set
            vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo { descriptorPool, 1, &descriptorSetLayout };
            const std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
            descriptorSet = descriptorSets.front();
            // Make image writable
            std::vector<vk::DescriptorImageInfo> imageInfos;
            vk::DescriptorImageInfo imageInfo {
                VK_NULL_HANDLE,
                storageImageView,
                vk::ImageLayout::eGeneral
            };
            device.updateDescriptorSets({ 
                {
                descriptorSet,
                0,
                (uint32_t) 0,
                (uint32_t) 1,
                vk::DescriptorType::eStorageImage,
                &imageInfo
                }
            }, {});
        };
        vk::CommandPool createCommandPool() {
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
        inline vk::CommandBuffer createCommandBuffer() {
            vk::CommandBufferAllocateInfo commandBufferAllocateInfo {
                commandPool,
                vk::CommandBufferLevel::ePrimary,
                1
            };
            const std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
            return commandBuffers.front();
        };
        void createSyncObjects() {
            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            vk::FenceCreateInfo fenceInfo {
                vk::FenceCreateFlagBits::eSignaled
            };
            imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphore = device.createSemaphore(semaphoreInfo);
            inFlightFence = device.createFence(fenceInfo);
            baseCommandsFinishedFence = device.createFence({});
        };
        void generateBaseCommands() {
            // Change image format 
            vk::ImageMemoryBarrier undefinedToGeneralBarrier(
                {}, 
                {}, 
                vk::ImageLayout::eUndefined, 
                vk::ImageLayout::eGeneral,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, storageImage,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer.begin(CmdBufferBeginInfo);
            commandBuffer.pipelineBarrier(
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
            commandBuffer.end();
            vk::SubmitInfo submitInfo { 
                0,			// Num Wait Semaphores
                nullptr,		// Wait Semaphores
                nullptr,		// Pipeline Stage Flags
                1,			// Num Command Buffers
                &commandBuffer };  // List of command buffers
            
            computeQueue.submit({ submitInfo }, baseCommandsFinishedFence);
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
            storageImage = createStorageImage();
            allocateStorageImage();
            storageImageView = createStorageImageView();
            createComputePipeline();
            commandPool = createCommandPool();
            commandBuffer = createCommandBuffer();
            createSyncObjects();
            generateBaseCommands();
            if (device.waitForFences(1, &baseCommandsFinishedFence, VK_TRUE, UINT64_MAX) == vk::Result::eErrorDeviceLost) throw std::runtime_error("device lost!");
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
        inline void drawFrame() {
            device.waitForFences(1, &inFlightFence, VK_TRUE, UINT64_MAX);
            device.resetFences(1, &inFlightFence);
            vk::ResultValue swapchainImageResult = device.acquireNextImageKHR(swapchain.swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE);
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
            commandBuffer.reset();
            vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer.begin(CmdBufferBeginInfo);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,	// Bind point
                pipelineLayout,				        // Pipeline Layout
                0,								    // First descriptor set
                { descriptorSet },					// List of descriptor sets
                {});								// Dynamic offsets
            commandBuffer.dispatch(WIDTH/16, (HEIGHT + HEIGHT % 16)/16, 1);
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
            commandBuffer.pipelineBarrier(
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
            commandBuffer.copyImage(storageImage, vk::ImageLayout::eGeneral, swapchainImage, vk::ImageLayout::eTransferDstOptimal, 1, &imageCopyRegion);
            commandBuffer.pipelineBarrier(
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
            commandBuffer.end();
            
            vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
            vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
            vk::PipelineStageFlags waitStageFlags[] = {vk::PipelineStageFlagBits::eComputeShader};
            vk::SubmitInfo submitInfo { 
                1,			// Num Wait Semaphores
                waitSemaphores,		// Wait Semaphores
                waitStageFlags,		// Pipeline Stage Flags
                1,			// Num Command Buffers
                &commandBuffer, // List of command buffers
                1,
                signalSemaphores
            };  
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;
            
            computeQueue.submit({ submitInfo }, inFlightFence);
            device.waitForFences({ inFlightFence },			// List of fences
                                true,				// Wait All
                                uint64_t(-1));		// Timeout
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
        };
        void loop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
            }
        };
    public:
        Application() {
            #if DEBUGMODE
            std::cout << "Creating application\n";
            #endif
        };
        void cleanup() {
            vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
            vkDestroyFence(device, inFlightFence, nullptr);
            vkDestroyFence(device, baseCommandsFinishedFence, nullptr);
            vkDestroyCommandPool(device, commandPool, nullptr);
            // TODO: contain storage image in wrapper
            device.destroyImage(storageImage);
            device.freeMemory(storageImageMemory);
            device.destroyImageView(storageImageView);
            // TODO: contain descriptor set in wrapper
            device.destroyDescriptorSetLayout(descriptorSetLayout);
            device.destroyDescriptorPool(descriptorPool);
            device.destroyPipeline(computePipeline);
            device.destroyPipelineLayout(pipelineLayout);
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