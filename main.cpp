#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <iostream>
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
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

// This is the application which is in essence a huge wrapper for the all the code that handles creating the window, keyboard inputs, etc.
class LuxosAppliation {
    GLFWwindow* window;
    vk::SurfaceKHR surface;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue computeQueue;
    vk::Queue presentQueue;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    vk::Format swapchainImageFormat;
    vk::Extent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;
    vk::Pipeline computePipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::CommandPool commandPool;
    vk::CommandBuffer commandBuffer;
    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;
    vk::Fence inFlightFence;
    vk::Image storageImage;
    vk::ImageView storageImageView;

    
    static std::vector<char> readFile(const std::string& filename) {
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
    }
    void allocateImage() {
        vk::ImageCreateInfo imageCreateInfo {
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

        };
        storageImage = device.createImage(imageCreateInfo);;
        // Allocate memory
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
        vk::MemoryAllocateInfo storageImageMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
		vk::DeviceMemory storageImageMemory = device.allocateMemory(storageImageMemoryAllocateInfo);
		device.bindImageMemory(storageImage, storageImageMemory, 0);
        // Create Image view
        vk::ImageViewCreateInfo createInfo {
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
        };
        storageImageView = device.createImageView(createInfo);
        // DEBUG: list out stats
        std::cout << "Memory Type Index    : " << memoryTypeIndex << "\n";
        std::cout << "Memory Heap Size     : " << memoryHeapSize / 1024 / 1024 / 1024 << " GB\n";
        std::cout << "Required Memory Size : " << memoryRequirements.size / 1024 / 1024 << " MB \n";
    }
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
        
        vk::Semaphore waitSemaphores[] = {imageAvailableSemaphore};
        
        vk::PipelineStageFlags waitStageFlags[] = {vk::PipelineStageFlagBits::eComputeShader};
		vk::SubmitInfo submitInfo { 
            0,			// Num Wait Semaphores
            nullptr,		// Wait Semaphores
            nullptr,		// Pipeline Stage Flags
            1,			// Num Command Buffers
            &commandBuffer };  // List of command buffers
        
        computeQueue.submit({ submitInfo }, inFlightFence);

    }
    void createComputePipeline() {
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
        vk::DescriptorSetLayout descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
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
        // Create descriptor pool
        vk::DescriptorPoolSize descriptorPoolSize { vk::DescriptorType::eStorageImage, 2 };
        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo {
            {}, 
            1,
            descriptorPoolSize
        };
        vk::DescriptorPool descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);
        // Allocate descriptor set
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo { descriptorPool, 1, &descriptorSetLayout };
        const std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);
        descriptorSet = descriptorSets.front();
        std::vector<vk::DescriptorImageInfo> imageInfos;
        vk::DescriptorImageInfo imageInfo {
            VK_NULL_HANDLE,
            storageImageView,
            vk::ImageLayout::eGeneral
        };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets { 
            {
            descriptorSet,
            0,
            (uint32_t) 0,
            (uint32_t) 1,
            vk::DescriptorType::eStorageImage,
            &imageInfo
            }
        };
		device.updateDescriptorSets(writeDescriptorSets, {});
    }
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = getQueueFamilyIndexes(physicalDevice);
        vk::CommandPoolCreateInfo poolInfo {
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            queueFamilyIndices.computeFamily.value()
        };
        device.createCommandPool(&poolInfo, nullptr, &commandPool);
    }
    void createCommandBuffer() {
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo {
            commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        };
		const std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
		commandBuffer = commandBuffers.front();
    }
    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
        renderFinishedSemaphore = device.createSemaphore(semaphoreInfo);
        inFlightFence = device.createFence(fenceInfo);
    }
    uint32_t rendered = 0;
    void drawFrame() {
        device.waitForFences(1, &inFlightFence, VK_TRUE, UINT64_MAX);
        device.resetFences(1, &inFlightFence);
        uint32_t imageIndex = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE).value;
        vk::Image swapchainImage = swapchainImages[imageIndex];
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
		commandBuffer.dispatch(WIDTH/16, HEIGHT/16, 1);
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
            {
                0,
                0,
                0
            },
            { WIDTH, 1060, 1 }
        };
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eByRegion,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &undefinedToOptimizedDestinationBarrier
        );
        commandBuffer.copyImage(storageImage, vk::ImageLayout::eGeneral, swapchainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, 1, &imageCopyRegion);
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
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
        

        vk::Semaphore waitSemaphores[] = {imageAvailableSemaphore};
        
        vk::PipelineStageFlags waitStageFlags[] = {vk::PipelineStageFlagBits::eComputeShader};
		vk::SubmitInfo submitInfo { 
            1,			// Num Wait Semaphores
            waitSemaphores,		// Wait Semaphores
            waitStageFlags,		// Pipeline Stage Flags
            1,			// Num Command Buffers
            &commandBuffer };  // List of command buffers
        
        computeQueue.submit({ submitInfo }, inFlightFence);
		device.waitForFences({ inFlightFence },			// List of fences
							 true,				// Wait All
							 uint64_t(-1));		// Timeout
        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 0;
        vk::SwapchainKHR swapchains[] = {swapchain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;
        if(presentQueue.presentKHR(&presentInfo) != vk::Result::eSuccess) std::cout << "failed\n";
        rendered++;
    }
    void initWindow () {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        if (window == nullptr) throw std::runtime_error("Failed to create window");
        glfwSetWindowPos(window, 0, 0);
    }
    void loop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
    }
    void createInstance() {
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
        vk::InstanceCreateInfo instanceCreateInfo {
            {},                      // Flags
            &applicationInfo,        // Application Info
            (uint32_t) validationLayers.size(), // Validation layer count
            validationLayers.data(), // Validaiton layers
            glfwExtensionCount,      // Extension count
            glfwExtensions,          // Extension names
            nullptr                  // pNext(idk what this is ngl)
        };

        if (!enableValidationLayers) instanceCreateInfo.enabledLayerCount = 0;
        instance = vk::createInstance(instanceCreateInfo);
        //DEBUG: Enumerate extensions available
        auto extensions = vk::enumerateInstanceExtensionProperties();
        std::cout << "Available extensions:\n";
        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    };
    void createSurface() {
	    auto p_surface = (VkSurfaceKHR) surface; // you can cast vk::* object to Vk* using a type cast
        if (glfwCreateWindowSurface(VkInstance(instance), window, nullptr, &p_surface) != VK_SUCCESS) throw std::runtime_error("failed to create window surface!");
        surface = p_surface;
    };
    struct QueueFamilyIndices {
        std::optional<uint32_t> computeFamily;
        std::optional<uint32_t> presentFamily;
        inline bool isComplete() {
            return computeFamily.has_value() && presentFamily.has_value();
        }
    };
    QueueFamilyIndices getQueueFamilyIndexes(vk::PhysicalDevice t_physicalDevice) {
        QueueFamilyIndices indices;
        auto queueFamilies = t_physicalDevice.getQueueFamilyProperties();
        int i = 0;
        for(const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) indices.computeFamily = i;
            if (t_physicalDevice.getSurfaceSupportKHR(i, surface)) indices.presentFamily = i;
            if (indices.isComplete()) break;
            i++;
        }
        return indices;
    };
    bool isDeviceSuitable(vk::PhysicalDevice t_physicalDevice) {
        if (!(getQueueFamilyIndexes(t_physicalDevice).isComplete())) return false;
        return true;
    };
    void choosePhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();
        if (devices.size() == 0) throw std::runtime_error("failed to find GPUs with Vulkan support!");
        for (const auto& device: devices) {
            // TODO: Implement device picking that gets the most preferred GPU
            if (isDeviceSuitable(device)){ 
                physicalDevice = device;
                break;
            }
        };
        if (physicalDevice == NULL) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        // DEBUG: List chosen device information
        vk::PhysicalDeviceProperties DeviceProps = physicalDevice.getProperties();
		const uint32_t ApiVersion = DeviceProps.apiVersion;
		vk::PhysicalDeviceLimits DeviceLimits = DeviceProps.limits;
		std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
		std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;
		std::cout << "Max Compute Shared Memory Size: " << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;
    };
    struct SwapchainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };
    SwapchainSupportDetails getSwapchainSupport(vk::PhysicalDevice t_physicalDevice) {
        SwapchainSupportDetails details;
        details.capabilities = t_physicalDevice.getSurfaceCapabilitiesKHR(surface);
        details.formats = t_physicalDevice.getSurfaceFormatsKHR(surface);
        details.presentModes = t_physicalDevice.getSurfacePresentModesKHR(surface);
        return details;
    };
    vk::SurfaceFormatKHR pickSurfaceFormat(std::vector<vk::SurfaceFormatKHR> &t_availableFormats) {
         for (const auto& availableFormat : t_availableFormats) {
            VkFormatProperties formatProperties;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8_UNORM, &formatProperties);
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return t_availableFormats[0];
    };
    vk::PresentModeKHR pickPresentMode(std::vector<vk::PresentModeKHR> t_availablePresentModes) {
        for (const auto& availablePresentMode : t_availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    };
    vk::Extent2D pickExtent(vk::SurfaceCapabilitiesKHR t_availableCapabilities) {
        if (t_availableCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return t_availableCapabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, t_availableCapabilities.minImageExtent.width, t_availableCapabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, t_availableCapabilities.minImageExtent.height, t_availableCapabilities.maxImageExtent.height);

            return actualExtent;
        }
    };
    void createSwapchain() {
        SwapchainSupportDetails swapchainSupportDetails = getSwapchainSupport(physicalDevice);
        // Pick the most optimal setup
        vk::SurfaceFormatKHR surfaceFormat = pickSurfaceFormat(swapchainSupportDetails.formats);
        vk::PresentModeKHR presentMode = pickPresentMode(swapchainSupportDetails.presentModes);
        vk::Extent2D extent = pickExtent(swapchainSupportDetails.capabilities);
        //Ensure that image count is the minimum count plus one to give some leeway(need to research this)
        uint32_t imageCount = swapchainSupportDetails.capabilities.minImageCount + 1;
        //Ensure that the image count is not more than the max. 0 is a special value meaning unlimited
        if (swapchainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapchainSupportDetails.capabilities.maxImageCount) {
            imageCount = swapchainSupportDetails.capabilities.maxImageCount;
        }
        std::cout << "The image count is " << imageCount << "\n";
        vk::SwapchainCreateInfoKHR createInfo {
            {},
            surface,                                               // Surface
            imageCount,                                            // Image count
            surfaceFormat.format,                                  // Image format
            surfaceFormat.colorSpace,                              // Color space
            extent,                                                // Image extent
            1,                                                     // Image array layers(1 unless doing VR basically)
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,              // Usage flags
            vk::SharingMode::eExclusive,                           // Sharing mode
            nullptr,                                               // Queue family indices
            swapchainSupportDetails.capabilities.currentTransform, // preTransform
            vk::CompositeAlphaFlagBitsKHR::eOpaque,                // Alpha blending
            presentMode,                                           // Present mode
            VK_FALSE,                                               // Clipped
            VK_NULL_HANDLE,                                        // Previous swapchain (has to do with resizing)
        };
        // Handle queue sharing
        QueueFamilyIndices indices = getQueueFamilyIndexes(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.computeFamily.value(), indices.presentFamily.value()};
        if (indices.computeFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        swapchain = device.createSwapchainKHR(createInfo);
        swapchainImages = device.getSwapchainImagesKHR(swapchain);
        swapchainImageFormat = surfaceFormat.format;
        swapchainExtent = extent;
    };
    void createImageViews() {
        int i = 0;
        for (vk::Image image : swapchainImages) {
            vk::ImageViewCreateInfo createInfo {
                {},
                image,
                vk::ImageViewType::e2D,
                swapchainImageFormat,
                vk::ComponentSwizzle {},
                vk::ImageSubresourceRange {
                    vk::ImageAspectFlagBits::eColor,
                    0,
                    1,
                    0,
                    1
                }
            };
            swapchainImageViews.push_back(device.createImageView(createInfo));
            i++;
        };
    };
    void createLogicalDevice() {
        // Prepare queue families
        QueueFamilyIndices indices = getQueueFamilyIndexes(physicalDevice);
        std::set<uint32_t> uniqueQueueFamilies = {indices.computeFamily.value(), indices.presentFamily.value()};
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos { };
		const float QueuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo { {}, queueFamily, 1, &QueuePriority };
            queueCreateInfos.push_back(queueCreateInfo);
        };
        //Device create info
        vk::DeviceCreateInfo deviceCreateInfo {
            {},
            (uint32_t) queueCreateInfos.size(),
            queueCreateInfos.data(), 
            (uint32_t) validationLayers.size(), // Validation layer count
            validationLayers.data(),            // Validaiton layers
            (uint32_t) deviceExtensions.size(), // Extension count
            deviceExtensions.data()             // Extensions
        };
        device = physicalDevice.createDevice(deviceCreateInfo);
        //Create queues
        computeQueue = device.getQueue(indices.computeFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    };
    void initVulkan() {
        createInstance();
        createSurface();
        //TODO: Allow multiple devices to be used at once
        choosePhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        allocateImage();
        createComputePipeline();
        createCommandPool();
        createCommandBuffer();
        generateBaseCommands();
        createSyncObjects();
    };
    public:
    void run() {
        std::cout << "Starting app\n";
        initWindow();
        initVulkan();
        loop();
    };
};
int main() {
    LuxosAppliation mainApp;
    try {
        mainApp.run();
        //app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}