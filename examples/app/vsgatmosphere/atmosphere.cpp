#include "atmosphere.h"
#include "atmosphere-tools.h"

#include <vsg/all.h>


namespace atmosphere {
// -----------------------------------------------------------------------------------------------------------------------------------

AtmosphereModel::AtmosphereModel(vsg::ref_ptr<vsg::Device> device, vsg::ref_ptr<vsg::PhysicalDevice> physicalDevice, vsg::ref_ptr<vsg::Options> options)
    : _device(device)
    , _physicalDevice(physicalDevice)
    , _options(options)
{
    compileSettings = vsg::ShaderCompileSettings::create();
}

// -----------------------------------------------------------------------------------------------------------------------------------

AtmosphereModel::~AtmosphereModel()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AtmosphereModel::initialize(int scatteringOrders)
{
    /*
    vsg::Names instanceExtensions;
    vsg::Names requestedLayers;
    vsg::Names deviceExtensions;

    if(compileSettings->generateDebugInfo)
        deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

    //instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    //requestedLayers.push_back("VK_LAYER_KHRONOS_validation");

    vsg::Names validatedNames = vsg::validateInstancelayerNames(requestedLayers);

    // get the physical device that supports the required compute queue
    auto instance = vsg::Instance::create(instanceExtensions, validatedNames);
    auto [physicalDevice, computeQueueFamily] = instance->getPhysicalDeviceAndQueueFamily(VK_QUEUE_COMPUTE_BIT);
    if (!physicalDevice || computeQueueFamily < 0)
    {
        //std::cout << "No vkPhysicalDevice available that supports compute." << std::endl;
        return;
    }

    // create the logical device with specified queue, layers and extensions
    vsg::QueueSettings queueSettings{vsg::QueueSetting{computeQueueFamily, {1.0}}};
    _device = vsg::Device::create(physicalDevice, queueSettings, validatedNames, deviceExtensions);
*/
    generateTextures();

    setupShaderConstants();

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    auto commandGraph = vsg::Commands::create();
    auto singlePass = vsg::Commands::create();
    auto multipleScattering = vsg::Commands::create();

    auto luminanceFromRadiance = bindLuminanceFromRadiance(vsg::mat4(), {kLambdaR, kLambdaG, kLambdaB});

    {
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{luminanceFromRadiance->setLayout}, vsg::PushConstantRanges{});
        auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, luminanceFromRadiance);
        bindDescriptorSet->slot = 2;
        singlePass->addChild(bindDescriptorSet);
    }

    {
        auto texturesSet = bindTransmittance();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_transmittance_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        singlePass->addChild(bindPipeline);
        singlePass->addChild(bindTextures);
        singlePass->addChild(vsg::Dispatch::create(uint32_t(ceil(float(transmittanceWidth) / float(numThreads))),
                                                     uint32_t(ceil(float(transmittanceHeight) / float(numThreads))), 1));
        singlePass->addChild(memoryBarrierCmd);
    }

    {
        auto texturesSet = bindDirectIrradiance();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_direct_irradiance_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        singlePass->addChild(bindPipeline);
        singlePass->addChild(bindTextures);
        singlePass->addChild(vsg::Dispatch::create(uint32_t(ceil(float(irradianceWidth) / float(numThreads))),
                                                      uint32_t(ceil(float(irradianceHeight) / float(numThreads))), 1));
        singlePass->addChild(memoryBarrierCmd);
    }

    {
        auto texturesSet = bindSingleScattering();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_single_scattering_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        singlePass->addChild(bindPipeline);
        singlePass->addChild(bindTextures);
        singlePass->addChild(vsg::Dispatch::create(uint32_t(scatteringWidth() / numThreads),
                                                     uint32_t(scatteringHeight() / numThreads),
                                                     uint32_t(scatteringDepth() / numThreads)));
        singlePass->addChild(memoryBarrierCmd);
    }

    auto orderL = orderLayout();
    auto orderPipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{orderL}, vsg::PushConstantRanges{});

    {
        auto texturesSet = bindScatteringDensity();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout, orderL}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_scattering_density_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        multipleScattering->addChild(bindPipeline);
        multipleScattering->addChild(bindTextures);
        multipleScattering->addChild(vsg::Dispatch::create(uint32_t(ceil(float(scatteringWidth()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringHeight()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringDepth()) / float(numThreads)))));
        multipleScattering->addChild(memoryBarrierCmd);
    }

    {
        auto texturesSet = bindIndirectIrradiance();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout, orderL}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_indirect_irradiance_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        multipleScattering->addChild(bindPipeline);
        multipleScattering->addChild(bindTextures);
        multipleScattering->addChild(vsg::Dispatch::create(uint32_t(ceil(float(irradianceWidth) / float(numThreads))),
                                                     uint32_t(ceil(float(irradianceHeight) / float(numThreads))), 1));
        multipleScattering->addChild(memoryBarrierCmd);
    }

    {
        auto texturesSet = bindMultipleScattering();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, luminanceFromRadiance->setLayout, orderL}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_multiple_scattering_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        multipleScattering->addChild(bindPipeline);
        multipleScattering->addChild(bindTextures);
        multipleScattering->addChild(vsg::Dispatch::create(uint32_t(ceil(float(scatteringWidth()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringHeight()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringDepth()) / float(numThreads)))));
        multipleScattering->addChild(memoryBarrierCmd);
    }

    commandGraph->addChild(singlePass);

    // Compute the 2nd, 3rd and 4th orderValue of scattering, in sequence.
    for (int order = 2; order <= scatteringOrders; ++order)
    {
        auto descriptor = vsg::DescriptorBuffer::create(vsg::intValue::create(order), 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        vsg::Descriptors descriptors{descriptor};
        auto set = vsg::DescriptorSet::create(orderL, descriptors);
        auto bindDescriptor = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, orderPipelineLayout, 2, set);
        bindDescriptor->slot = 3;
        commandGraph->addChild(bindDescriptor);

        commandGraph->addChild(multipleScattering);
    }

    // compile the Vulkan objects
    auto compileTraversal = vsg::CompileTraversal::create(_device);
    auto context = compileTraversal->contexts.front();

    commandGraph->accept(*compileTraversal);

    int computeQueueFamily = _physicalDevice->getQueueFamily(VK_QUEUE_COMPUTE_BIT);

    context->commandPool = vsg::CommandPool::create(_device, computeQueueFamily);

    auto fence = vsg::Fence::create(_device);
    auto computeQueue = _device->getQueue(computeQueueFamily);

    vsg::submitCommandsToQueue(context->commandPool, fence, 100000000000, computeQueue, [&](vsg::CommandBuffer& commandBuffer) {
        commandGraph->record(commandBuffer);
    });
    /*
    transmittanceData = mapData(_transmittanceTexture->imageView, transmittanceWidth, transmittanceHeight);
    scatteringData = mapData(_scatteringTextureSndOrder->imageView, scatteringWidth(), scatteringWidth(), scatteringDepth());
    irradianceData = mapData(_irradianceTexture->imageView, irradianceWidth, irradianceHeight);
    mieScatteringData = mapData(_singleMieScatteringTexture->imageView, scatteringWidth(), scatteringWidth(), scatteringDepth());
    */
}

vsg::ref_ptr<vsg::CommandGraph> AtmosphereModel::createCubeMapGraph(vsg::ref_ptr<vsg::Value<RuntimeSettings> > settings)
{
    auto computeShader = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/reflection_map.glsl", _options);
    computeShader->module->hints = compileSettings;

    assignRenderConstants();
    computeShader->specializationConstants = _constants;

    // binding 0 source buffer
    // binding 1 image2d

    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto settingsBuffer = vsg::DescriptorBuffer::create(settings, 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto scatteringTexture = vsg::DescriptorImage::create(_scatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto singleMieTexture = vsg::DescriptorImage::create(_singleMieScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto cubemap = vsg::DescriptorImage::create(_cubeMap, 5, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});

    vsg::Descriptors descriptors{settingsBuffer, transmittanceTexture, irradianceTexture, scatteringTexture, singleMieTexture, cubemap};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeShader);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto preCopyBarrier = vsg::ImageMemoryBarrier::create();
    preCopyBarrier->srcAccessMask = 0;
    preCopyBarrier->dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    preCopyBarrier->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    preCopyBarrier->newLayout = VK_IMAGE_LAYOUT_GENERAL;
    preCopyBarrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preCopyBarrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preCopyBarrier->image = _cubeMap->imageView->image;
    preCopyBarrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    preCopyBarrier->subresourceRange.baseArrayLayer = 0;
    preCopyBarrier->subresourceRange.layerCount = 6;
    preCopyBarrier->subresourceRange.levelCount = 1;
    preCopyBarrier->subresourceRange.baseMipLevel = 0;

    auto preCopyBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, preCopyBarrier);

    auto postCopyBarrier = vsg::ImageMemoryBarrier::create();
    postCopyBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postCopyBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    postCopyBarrier->oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postCopyBarrier->newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postCopyBarrier->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postCopyBarrier->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    postCopyBarrier->image = _cubeMap->imageView->image;
    postCopyBarrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    postCopyBarrier->subresourceRange.baseArrayLayer = 0;
    postCopyBarrier->subresourceRange.layerCount = 6;
    postCopyBarrier->subresourceRange.levelCount = 1;
    postCopyBarrier->subresourceRange.baseMipLevel = 0;

    auto postCopyBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, postCopyBarrier);

    int computeQueueFamily = _physicalDevice->getQueueFamily(VK_QUEUE_COMPUTE_BIT);
    auto compute_commandGraph = vsg::CommandGraph::create(_device, computeQueueFamily);

    compute_commandGraph->addChild(preCopyBarrierCmd);
    compute_commandGraph->addChild(bindPipeline);
    compute_commandGraph->addChild(bindDescriptorSet);
    auto workgroups = uint32_t(ceil(float(cubeSize) / float(numViewerThreads)));
    compute_commandGraph->addChild(vsg::Dispatch::create(workgroups, workgroups, 6));
    compute_commandGraph->addChild(postCopyBarrierCmd);

    return compute_commandGraph;
}

void AtmosphereModel::generateTextures()
{
    _transmittanceTexture = generate2D(transmittanceWidth, transmittanceHeight);

    _irradianceTexture = generate2D(irradianceWidth, irradianceHeight, true);
    _deltaIrradianceTexture = generate2D(irradianceWidth, irradianceHeight);

    auto width = scatteringWidth();
    auto height = scatteringHeight();
    auto depth = scatteringDepth();

    _deltaRayleighScatteringTexture = generate3D(width, height, depth);
    _deltaMieScatteringTexture = generate3D(width, height, depth);
    _scatteringTexture = generate3D(width, height, depth, true);
    _singleMieScatteringTexture = generate3D(width, height, depth, true);

    _deltaScatteringDensityTexture = generate3D(width, height, depth);
    _deltaMultipleScatteringTexture = generate3D(width, height, depth);

    _cubeMap = generateCubemap(cubeSize);
}

vsg::ref_ptr<vsg::ImageInfo> AtmosphereModel::generate2D(uint32_t width, uint32_t height, bool init)
{
    vsg::ref_ptr<vsg::Image> image = vsg::Image::create();
    image->usage |= (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    image->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image->mipLevels = 1;
    image->extent = VkExtent3D{width, height, 1};
    image->imageType = VK_IMAGE_TYPE_2D;
    image->arrayLayers = 1;

    if(init)
        image->data = vsg::vec4Array2D::create(width, height, vsg::vec4{0.0f, 0.0f, 0.0f, 1.0f}, vsg::Data::Properties{VK_FORMAT_R32G32B32A32_SFLOAT});

    image->compile(_device);
    image->allocateAndBindMemory(_device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    auto imageView = vsg::ImageView::create(image, VK_IMAGE_ASPECT_COLOR_BIT);
    auto sampler = vsg::Sampler::create();
    return vsg::ImageInfo::create(sampler, imageView, VK_IMAGE_LAYOUT_GENERAL);

}

vsg::ref_ptr<vsg::ImageInfo> AtmosphereModel::generate3D(uint32_t width, uint32_t height, uint32_t depth, bool init)
{
    vsg::ref_ptr<vsg::Image> image = vsg::Image::create();
    image->usage |= (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    image->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image->mipLevels = 1;
    image->extent = VkExtent3D{width, height, depth};
    image->imageType = VK_IMAGE_TYPE_3D;
    image->arrayLayers = 1;

    if(init)
        image->data = vsg::vec4Array3D::create(width, height, depth, vsg::vec4{0.0f, 0.0f, 0.0f, 1.0f}, vsg::Data::Properties{VK_FORMAT_R32G32B32A32_SFLOAT});

    image->compile(_device);
    image->allocateAndBindMemory(_device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    auto imageView = vsg::ImageView::create(image, VK_IMAGE_ASPECT_COLOR_BIT);
    auto sampler = vsg::Sampler::create();
    return vsg::ImageInfo::create(sampler, imageView, VK_IMAGE_LAYOUT_GENERAL);

}

vsg::ref_ptr<vsg::ImageInfo> AtmosphereModel::generateCubemap(uint32_t size)
{
    vsg::ref_ptr<vsg::Image> image = vsg::Image::create();
    image->usage |= (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    image->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image->mipLevels = 1;
    image->extent = VkExtent3D{size, size, 1};
    image->imageType = VK_IMAGE_TYPE_2D;
    image->arrayLayers = 6;

    //image->compile(_device);
    //image->allocateAndBindMemory(_device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    auto imageView = vsg::ImageView::create(image, VK_IMAGE_ASPECT_COLOR_BIT);
    imageView->viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    auto sampler = vsg::Sampler::create();
    return vsg::ImageInfo::create(sampler, imageView, VK_IMAGE_LAYOUT_GENERAL);
}

vsg::ref_ptr<vsg::BindComputePipeline> AtmosphereModel::bindCompute(const vsg::Path& filename, vsg::ref_ptr<vsg::PipelineLayout> pipelineLayout) const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", filename, _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);

    return vsg::BindComputePipeline::create(pipeline);;
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindTransmittance() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto writeTexture = vsg::DescriptorImage::create(_transmittanceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    vsg::Descriptors descriptors{writeTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindDirectIrradiance() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
         {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto deltaTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{deltaTexture, transmittanceTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindSingleScattering() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
         {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto rayTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto readTexture = vsg::DescriptorImage::create(_scatteringTexture, 2, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto writeTexture = vsg::DescriptorImage::create(_singleMieScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{rayTexture, mieTexture, readTexture, writeTexture, transmittanceTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindScatteringDensity() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto scatteringTexture = vsg::DescriptorImage::create(_deltaScatteringDensityTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto rayTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto multipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto irradianceTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{scatteringTexture, transmittanceTexture, rayTexture, mieTexture, multipleTexture, irradianceTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindIndirectIrradiance() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto deltaIrradianceTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto singleTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto multipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    //auto orderValueBuffer = vsg::DescriptorBuffer::create(orderValue, 6);

    vsg::Descriptors descriptors{deltaIrradianceTexture, irradianceTexture, singleTexture,
                mieTexture, multipleTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindMultipleScattering() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);


    auto deltaMultipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto scatteringTexture = vsg::DescriptorImage::create(_scatteringTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto deltaDensityTexture = vsg::DescriptorImage::create(_deltaScatteringDensityTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{deltaMultipleTexture, scatteringTexture, transmittanceTexture, deltaDensityTexture};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSet> AtmosphereModel::bindLuminanceFromRadiance(const vsg::mat4& value, const vsg::vec3& lambdas) const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    Params params;
    params.solar_irradiance = toVector(waveLengths, solarIrradiance, lambdas, 1.0);
    params.rayleigh_scattering = toVector(waveLengths, rayleighScattering, lambdas, lengthUnitInMeters);
    params.mie_scattering = toVector(waveLengths, mieScattering, lambdas, lengthUnitInMeters);
    params.mie_extinction = toVector(waveLengths, mieExtinction, lambdas, lengthUnitInMeters);
    params.absorption_extinction = toVector(waveLengths, absorptionExtinction, lambdas, lengthUnitInMeters);
    params.ground_albedo = toVector(waveLengths, groundAlbedo, lambdas, 1.0);

    auto matrix = vsg::DescriptorBuffer::create(vsg::mat4Value::create(value), 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto buffer = vsg::DescriptorBuffer::create(vsg::Value<Params>::create(params), 1, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    vsg::Descriptors descriptors{matrix, buffer};
    return vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
}

vsg::ref_ptr<vsg::DescriptorSetLayout> AtmosphereModel::orderLayout() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    return vsg::DescriptorSetLayout::create(descriptorBindings);
}

// -----------------------------------------------------------------------------------------------------------------------------------
/*
void AtmosphereModel::bind_rendering_uniforms(dw::Program* program)
{
    if (program->set_uniform("transmittance_texture", 0))
        m_transmittance_texture->bind(0);

    if (program->set_uniform("scattering_texture", 1))
        m_scattering_texture->bind(1);

    if (program->set_uniform("irradiance_texture", 2))
        m_irradiance_texture->bind(2);

    if (!m_combine_scattering_textures)
    {
        if (program->set_uniform("single_mie_scattering_texture", 3))
            m_optional_single_mie_scattering_texture->bind(3);
    }

    program->set_uniform("TRANSMITTANCE_TEXTURE_WIDTH", CONSTANTS::TRANSMITTANCE_WIDTH);
    program->set_uniform("TRANSMITTANCE_TEXTURE_HEIGHT", CONSTANTS::TRANSMITTANCE_HEIGHT);
    program->set_uniform("SCATTERING_TEXTURE_R_SIZE", CONSTANTS::SCATTERING_R);
    program->set_uniform("SCATTERING_TEXTURE_MU_SIZE", CONSTANTS::SCATTERING_MU);
    program->set_uniform("SCATTERING_TEXTURE_MU_S_SIZE", CONSTANTS::SCATTERING_MU_S);
    program->set_uniform("SCATTERING_TEXTURE_NU_SIZE", CONSTANTS::SCATTERING_NU);
    program->set_uniform("SCATTERING_TEXTURE_WIDTH", CONSTANTS::SCATTERING_WIDTH);
    program->set_uniform("SCATTERING_TEXTURE_HEIGHT", CONSTANTS::SCATTERING_HEIGHT);
    program->set_uniform("SCATTERING_TEXTURE_DEPTH", CONSTANTS::SCATTERING_DEPTH);
    program->set_uniform("IRRADIANCE_TEXTURE_WIDTH", CONSTANTS::IRRADIANCE_WIDTH);
    program->set_uniform("IRRADIANCE_TEXTURE_HEIGHT", CONSTANTS::IRRADIANCE_HEIGHT);

    program->set_uniform("sun_angular_radius", (float)m_sun_angular_radius);
    program->set_uniform("bottom_radius", (float)(m_bottom_radius/ m_length_unit_in_meters));
    program->set_uniform("top_radius", (float)(m_top_radius / m_length_unit_in_meters));
    program->set_uniform("mie_phase_function_g", (float)m_mie_phase_function_g);
    program->set_uniform("mu_s_min", (float)cos(m_max_sun_zenith_angle));

    glm::vec3 sky_spectral_radiance_to_luminance, sun_spectral_radiance_to_luminance;
    sky_sun_radiance_to_luminance(sky_spectral_radiance_to_luminance, sun_spectral_radiance_to_luminance);

    program->set_uniform("SKY_SPECTRAL_RADIANCE_TO_LUMINANCE", sky_spectral_radiance_to_luminance);
    program->set_uniform("SUN_SPECTRAL_RADIANCE_TO_LUMINANCE", sun_spectral_radiance_to_luminance);

    double lambdas[] = { kLambdaR, kLambdaG, kLambdaB };

    glm::vec3 solar_irradiance = to_vector(m_wave_lengths, m_solar_irradiance, lambdas, 1.0);
    program->set_uniform("solar_irradiance", solar_irradiance);

    glm::vec3 rayleigh_scattering = to_vector(m_wave_lengths, m_rayleigh_scattering, lambdas, m_length_unit_in_meters);
    program->set_uniform("rayleigh_scattering", rayleigh_scattering);

    glm::vec3 mie_scattering = to_vector(m_wave_lengths, m_mie_scattering, lambdas, m_length_unit_in_meters);
    program->set_uniform("mie_scattering", mie_scattering);
}
*/
// -----------------------------------------------------------------------------------------------------------------------------------

void AtmosphereModel::setupShaderConstants()
{
    vsg::vec3 lambdas(kLambdaR, kLambdaG, kLambdaB);

    auto solar_irradiance = toVector(waveLengths, solarIrradiance, lambdas, 1.0);
    auto rayleigh_scattering = toVector(waveLengths, rayleighScattering, lambdas, lengthUnitInMeters);
    auto mie_scattering = toVector(waveLengths, mieScattering, lambdas, lengthUnitInMeters);
    auto mie_extinction = toVector(waveLengths, mieExtinction, lambdas, lengthUnitInMeters);
    auto absorption_extinction = toVector(waveLengths, absorptionExtinction, lambdas, lengthUnitInMeters);
    auto ground_albedo = toVector(waveLengths, groundAlbedo, lambdas, 1.0);

    _constants = {
        {0, vsg::intValue::create(numThreads)},
        {1, vsg::intValue::create(numViewerThreads)},
        {2, vsg::intValue::create(cubeSize)},

        {10, vsg::intValue::create(transmittanceWidth)},
        {11, vsg::intValue::create(transmittanceHeight)},

        {12, vsg::intValue::create(scaterringR)},
        {13, vsg::intValue::create(scaterringMU)},
        {14, vsg::intValue::create(scaterringMU_S)},
        {15, vsg::intValue::create(scaterringNU)},

        {16, vsg::intValue::create(irradianceWidth)},
        {17, vsg::intValue::create(irradianceHeight)},

        {18, vsg::floatValue::create(rayleigh_scattering.x)},
        {19, vsg::floatValue::create(rayleigh_scattering.y)},
        {20, vsg::floatValue::create(rayleigh_scattering.z)},

        {21, vsg::floatValue::create(mie_scattering.x)},
        {22, vsg::floatValue::create(mie_scattering.y)},
        {23, vsg::floatValue::create(mie_scattering.z)},

        {24, vsg::floatValue::create(absorption_extinction.x)},
        {25, vsg::floatValue::create(absorption_extinction.y)},
        {26, vsg::floatValue::create(absorption_extinction.z)},

        {27, vsg::floatValue::create(solar_irradiance.x)},
        {28, vsg::floatValue::create(solar_irradiance.y)},
        {29, vsg::floatValue::create(solar_irradiance.z)},

        {30, vsg::floatValue::create(mie_extinction.x)},
        {31, vsg::floatValue::create(mie_extinction.y)},
        {32, vsg::floatValue::create(mie_extinction.z)},

        {33, vsg::floatValue::create(ground_albedo.x)},
        {34, vsg::floatValue::create(ground_albedo.y)},
        {35, vsg::floatValue::create(ground_albedo.z)},

        {36, vsg::floatValue::create(static_cast<float>(sunAngularRadius))},
        {37, vsg::floatValue::create(static_cast<float>(bottomRadius / lengthUnitInMeters))},
        {38, vsg::floatValue::create(static_cast<float>(topRadius / lengthUnitInMeters))},
        {39, vsg::floatValue::create(static_cast<float>(miePhaseFunction_g))},
        {40, vsg::floatValue::create(static_cast<float>(std::cos(maxSunZenithAngle)))},
/*
        {41, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {42, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {43, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},

        {44, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {45, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {46, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},
*/
        {47, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.width / lengthUnitInMeters))},
        {48, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.exp_term))},
        {49, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.exp_scale * lengthUnitInMeters))},
        {50, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.linear_term * lengthUnitInMeters))},
        {51, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.constant_term))},

        {52, vsg::floatValue::create(static_cast<float>(mieDensityLayer.width / lengthUnitInMeters))},
        {53, vsg::floatValue::create(static_cast<float>(mieDensityLayer.exp_term))},
        {54, vsg::floatValue::create(static_cast<float>(mieDensityLayer.exp_scale * lengthUnitInMeters))},
        {55, vsg::floatValue::create(static_cast<float>(mieDensityLayer.linear_term * lengthUnitInMeters))},
        {56, vsg::floatValue::create(static_cast<float>(mieDensityLayer.constant_term))},

        {57, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.width / lengthUnitInMeters))},
        {58, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.exp_term))},
        {59, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.exp_scale * lengthUnitInMeters))},
        {60, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.linear_term * lengthUnitInMeters))},
        {61, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.constant_term))},

        {62, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.width / lengthUnitInMeters))},
        {63, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.exp_term))},
        {64, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.exp_scale * lengthUnitInMeters))},
        {65, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.linear_term * lengthUnitInMeters))},
        {66, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.constant_term))}
    };
}

void AtmosphereModel::assignRenderConstants()
{
    auto SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = computeSpectralRadianceToLuminanceFactors(waveLengths, solarIrradiance, -3);
    auto SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = computeSpectralRadianceToLuminanceFactors(waveLengths, solarIrradiance, 0);

    vsg::vec3 lambdas(kLambdaR, kLambdaG, kLambdaB);

    auto solar_irradiance = toVector(waveLengths, solarIrradiance, lambdas, 1.0);
    auto rayleigh_scattering = toVector(waveLengths, rayleighScattering, lambdas, lengthUnitInMeters);
    auto mie_scattering = toVector(waveLengths, mieScattering, lambdas, lengthUnitInMeters);
    auto mie_extinction = toVector(waveLengths, mieExtinction, lambdas, lengthUnitInMeters);
    auto absorption_extinction = toVector(waveLengths, absorptionExtinction, lambdas, lengthUnitInMeters);
    auto ground_albedo = toVector(waveLengths, groundAlbedo, lambdas, 1.0);

    _constants = {
        {0, vsg::intValue::create(numThreads)},
        {1, vsg::intValue::create(numViewerThreads)},
        {2, vsg::intValue::create(cubeSize)},

        {10, vsg::intValue::create(transmittanceWidth)},
        {11, vsg::intValue::create(transmittanceHeight)},

        {12, vsg::intValue::create(scaterringR)},
        {13, vsg::intValue::create(scaterringMU)},
        {14, vsg::intValue::create(scaterringMU_S)},
        {15, vsg::intValue::create(scaterringNU)},

        {16, vsg::intValue::create(irradianceWidth)},
        {17, vsg::intValue::create(irradianceHeight)},

        {18, vsg::floatValue::create(rayleigh_scattering.x)},
        {19, vsg::floatValue::create(rayleigh_scattering.y)},
        {20, vsg::floatValue::create(rayleigh_scattering.z)},

        {21, vsg::floatValue::create(mie_scattering.x)},
        {22, vsg::floatValue::create(mie_scattering.y)},
        {23, vsg::floatValue::create(mie_scattering.z)},

        {24, vsg::floatValue::create(absorption_extinction.x)},
        {25, vsg::floatValue::create(absorption_extinction.y)},
        {26, vsg::floatValue::create(absorption_extinction.z)},

        {27, vsg::floatValue::create(solar_irradiance.x)},
        {28, vsg::floatValue::create(solar_irradiance.y)},
        {29, vsg::floatValue::create(solar_irradiance.z)},

        {30, vsg::floatValue::create(mie_extinction.x)},
        {31, vsg::floatValue::create(mie_extinction.y)},
        {32, vsg::floatValue::create(mie_extinction.z)},

        {33, vsg::floatValue::create(ground_albedo.x)},
        {34, vsg::floatValue::create(ground_albedo.y)},
        {35, vsg::floatValue::create(ground_albedo.z)},

        {36, vsg::floatValue::create(static_cast<float>(sunAngularRadius))},
        {37, vsg::floatValue::create(static_cast<float>(bottomRadius / lengthUnitInMeters))},
        {38, vsg::floatValue::create(static_cast<float>(topRadius / lengthUnitInMeters))},
        {39, vsg::floatValue::create(static_cast<float>(miePhaseFunction_g))},
        {40, vsg::floatValue::create(static_cast<float>(std::cos(maxSunZenithAngle)))},

        {41, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {42, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {43, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},

        {44, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {45, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {46, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},

        {47, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.width / lengthUnitInMeters))},
        {48, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.exp_term))},
        {49, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.exp_scale * lengthUnitInMeters))},
        {50, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.linear_term * lengthUnitInMeters))},
        {51, vsg::floatValue::create(static_cast<float>(rayleighDensityLayer.constant_term))},

        {52, vsg::floatValue::create(static_cast<float>(mieDensityLayer.width / lengthUnitInMeters))},
        {53, vsg::floatValue::create(static_cast<float>(mieDensityLayer.exp_term))},
        {54, vsg::floatValue::create(static_cast<float>(mieDensityLayer.exp_scale * lengthUnitInMeters))},
        {55, vsg::floatValue::create(static_cast<float>(mieDensityLayer.linear_term * lengthUnitInMeters))},
        {56, vsg::floatValue::create(static_cast<float>(mieDensityLayer.constant_term))},

        {57, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.width / lengthUnitInMeters))},
        {58, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.exp_term))},
        {59, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.exp_scale * lengthUnitInMeters))},
        {60, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.linear_term * lengthUnitInMeters))},
        {61, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer0.constant_term))},

        {62, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.width / lengthUnitInMeters))},
        {63, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.exp_term))},
        {64, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.exp_scale * lengthUnitInMeters))},
        {65, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.linear_term * lengthUnitInMeters))},
        {66, vsg::floatValue::create(static_cast<float>(absorptionDensityLayer1.constant_term))}
    };

}

vsg::ref_ptr<vsg::Data> AtmosphereModel::mapData(vsg::ref_ptr<vsg::ImageView> view, uint32_t width, uint32_t height)
{
    auto buffer = view->image->getDeviceMemory(_device->deviceID);
    auto offset = view->image->getMemoryOffset(_device->deviceID);

    return vsg::MappedData<vsg::vec4Array2D>::create(buffer, offset, 0, vsg::Data::Properties{VK_FORMAT_R32G32B32A32_SFLOAT}, width, height);
}

vsg::ref_ptr<vsg::Data> AtmosphereModel::mapData(vsg::ref_ptr<vsg::ImageView> view, uint32_t width, uint32_t height, uint32_t depth)
{
    auto buffer = view->image->getDeviceMemory(_device->deviceID);
    auto offset = view->image->getMemoryOffset(_device->deviceID);

    return vsg::MappedData<vsg::vec4Array3D>::create(buffer, offset, 0, vsg::Data::Properties{VK_FORMAT_R32G32B32A32_SFLOAT}, width, height, depth);
}

// -----------------------------------------------------------------------------------------------------------------------------------
/*
void AtmosphereModel::bind_density_layer(dw::Program* program, DensityProfileLayer* layer)
{
    program->set_uniform(layer->name + "_width", (float)(layer->width / m_length_unit_in_meters));
    program->set_uniform(layer->name + "_exp_term", (float)layer->exp_term);
    program->set_uniform(layer->name + "_exp_scale", (float)(layer->exp_scale * m_length_unit_in_meters));
    program->set_uniform(layer->name + "_linear_term", (float)(layer->linear_term * m_length_unit_in_meters));
    program->set_uniform(layer->name + "_constant_term", (float)layer->constant_term);
}


void AtmosphereModel::precompute(TextureBuffer* buffer, double* lambdas, double* luminance_from_radiance, bool blend, int num_scattering_orderValues)
{
    int BLEND = blend ? 1 : 0;
    int NUM_THREADS = CONSTANTS::NUM_THREADS;

    // ------------------------------------------------------------------
    // Compute Transmittance
    // ------------------------------------------------------------------

    m_transmittance_program->use();

    bind_compute_uniforms(m_transmittance_program, lambdas, luminance_from_radiance);

    // Compute the transmittance, and store it in transmittance_texture
    buffer->m_transmittance_array[WRITE]->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_transmittance_array[WRITE]->internal_format());

    m_transmittance_program->set_uniform("blend", glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

    GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::TRANSMITTANCE_WIDTH / NUM_THREADS, CONSTANTS::TRANSMITTANCE_HEIGHT / NUM_THREADS, 1));
    GL_CHECK_ERROR(glFinish());

    swap(buffer->m_transmittance_array);

    // ------------------------------------------------------------------
    // Compute Direct Irradiance
    // ------------------------------------------------------------------

    m_direct_irradiance_program->use();

    bind_compute_uniforms(m_direct_irradiance_program, lambdas, luminance_from_radiance);

    // Compute the direct irradiance, store it in delta_irradiance_texture and,
    // depending on 'blend', either initialize irradiance_texture_ with zeros or
    // leave it unchanged (we don't want the direct irradiance in
    // irradiance_texture_, but only the irradiance from the sky).
    buffer->m_irradiance_array[READ]->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_irradiance_array[READ]->internal_format());
    buffer->m_irradiance_array[WRITE]->bind_image(1, 0, 0, GL_READ_WRITE, buffer->m_irradiance_array[WRITE]->internal_format());
    buffer->m_delta_irradiance_texture->bind_image(2, 0, 0, GL_READ_WRITE, buffer->m_delta_irradiance_texture->internal_format());

    if (m_direct_irradiance_program->set_uniform("transmittance", 3))
        buffer->m_transmittance_array[READ]->bind(3);

    m_direct_irradiance_program->set_uniform("blend", glm::vec4(0.0f, BLEND, 0.0f, 0.0f));

    GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::IRRADIANCE_WIDTH / NUM_THREADS, CONSTANTS::IRRADIANCE_HEIGHT / NUM_THREADS, 1));
    GL_CHECK_ERROR(glFinish());

    swap(buffer->m_irradiance_array);

    // ------------------------------------------------------------------
    // Compute Single Scattering
    // ------------------------------------------------------------------

    m_single_scattering_program->use();

    bind_compute_uniforms(m_single_scattering_program, lambdas, luminance_from_radiance);

    // Compute the rayleigh and mie single scattering, store them in
    // delta_rayleigh_scattering_texture and delta_mie_scattering_texture, and
    // either store them or accumulate them in scattering_texture_ and
    // optional_single_mie_scattering_texture_.
    buffer->m_delta_rayleigh_scattering_texture->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_delta_rayleigh_scattering_texture->internal_format());
    buffer->m_delta_mie_scattering_texture->bind_image(1, 0, 0, GL_READ_WRITE, buffer->m_delta_mie_scattering_texture->internal_format());
    buffer->m_scattering_array[READ]->bind_image(2, 0, 0, GL_READ_WRITE, buffer->m_scattering_array[READ]->internal_format());
    buffer->m_scattering_array[WRITE]->bind_image(3, 0, 0, GL_READ_WRITE, buffer->m_scattering_array[WRITE]->internal_format());
    buffer->m_optional_single_mie_scattering_array[READ]->bind_image(4, 0, 0, GL_READ_WRITE, buffer->m_optional_single_mie_scattering_array[READ]->internal_format());
    buffer->m_optional_single_mie_scattering_array[WRITE]->bind_image(5, 0, 0, GL_READ_WRITE, buffer->m_optional_single_mie_scattering_array[WRITE]->internal_format());

    if (m_single_scattering_program->set_uniform("transmittance", 6))
        buffer->m_transmittance_array[READ]->bind(6);

    m_single_scattering_program->set_uniform("blend", glm::vec4(0.0f, 0.0f, BLEND, BLEND));

    for (int layer = 0; layer < CONSTANTS::SCATTERING_DEPTH; ++layer)
    {
        m_single_scattering_program->set_uniform("layer", layer);

        GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::SCATTERING_WIDTH / NUM_THREADS, CONSTANTS::SCATTERING_HEIGHT / NUM_THREADS, 1));
        GL_CHECK_ERROR(glFinish());
    }

    swap(buffer->m_scattering_array);
    swap(buffer->m_optional_single_mie_scattering_array);


    // Compute the 2nd, 3rd and 4th orderValue of scattering, in sequence.
    for (int scattering_orderValue = 2; scattering_orderValue <= num_scattering_orderValues; ++scattering_orderValue)
    {
        // ------------------------------------------------------------------
        // Compute Scattering Density
        // ------------------------------------------------------------------

        m_scattering_density_program->use();

        bind_compute_uniforms(m_scattering_density_program, lambdas, luminance_from_radiance);

        // Compute the scattering density, and store it in
        // delta_scattering_density_texture.
        buffer->m_delta_scattering_density_texture->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_delta_scattering_density_texture->internal_format());

        if (m_scattering_density_program->set_uniform("transmittance", 1))
            buffer->m_transmittance_array[READ]->bind(1);

        if (m_scattering_density_program->set_uniform("single_rayleigh_scattering", 2))
            buffer->m_delta_rayleigh_scattering_texture->bind(2);

        if (m_scattering_density_program->set_uniform("single_mie_scattering", 3))
            buffer->m_delta_mie_scattering_texture->bind(3);

        if (m_scattering_density_program->set_uniform("multiple_scattering", 4))
            buffer->m_delta_multiple_scattering_texture->bind(4);

        if (m_scattering_density_program->set_uniform("irradiance", 5))
            buffer->m_delta_irradiance_texture->bind(5);

        m_scattering_density_program->set_uniform("scattering_orderValue", scattering_orderValue);
        m_scattering_density_program->set_uniform("blend", glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

        for (int layer = 0; layer < CONSTANTS::SCATTERING_DEPTH; ++layer)
        {
            m_scattering_density_program->set_uniform("layer", layer);
            GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::SCATTERING_WIDTH / NUM_THREADS, CONSTANTS::SCATTERING_HEIGHT / NUM_THREADS, 1));
            GL_CHECK_ERROR(glFinish());
        }

        // ------------------------------------------------------------------
        // Compute Indirect Irradiance
        // ------------------------------------------------------------------

        m_indirect_irradiance_program->use();

        bind_compute_uniforms(m_indirect_irradiance_program, lambdas, luminance_from_radiance);

        // Compute the indirect irradiance, store it in delta_irradiance_texture and
        // accumulate it in irradiance_texture_.
        buffer->m_delta_irradiance_texture->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_delta_irradiance_texture->internal_format());
        buffer->m_irradiance_array[READ]->bind_image(1, 0, 0, GL_READ_WRITE, buffer->m_irradiance_array[READ]->internal_format());
        buffer->m_irradiance_array[WRITE]->bind_image(2, 0, 0, GL_READ_WRITE, buffer->m_irradiance_array[WRITE]->internal_format());

        if (m_indirect_irradiance_program->set_uniform("single_rayleigh_scattering", 3))
            buffer->m_delta_rayleigh_scattering_texture->bind(3);

        if (m_indirect_irradiance_program->set_uniform("single_mie_scattering", 4))
            buffer->m_delta_mie_scattering_texture->bind(4);

        if (m_indirect_irradiance_program->set_uniform("multiple_scattering", 5))
            buffer->m_delta_multiple_scattering_texture->bind(5);

        m_indirect_irradiance_program->set_uniform("scattering_orderValue", scattering_orderValue - 1);
        m_indirect_irradiance_program->set_uniform("blend", glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

        GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::IRRADIANCE_WIDTH / NUM_THREADS, CONSTANTS::IRRADIANCE_HEIGHT / NUM_THREADS, 1));
        GL_CHECK_ERROR(glFinish());

        swap(buffer->m_irradiance_array);

        // ------------------------------------------------------------------
        // Compute Multiple Scattering
        // ------------------------------------------------------------------

        m_multiple_scattering_program->use();

        bind_compute_uniforms(m_multiple_scattering_program, lambdas, luminance_from_radiance);

        // Compute the multiple scattering, store it in
        // delta_multiple_scattering_texture, and accumulate it in
        // scattering_texture_.
        buffer->m_delta_multiple_scattering_texture->bind_image(0, 0, 0, GL_READ_WRITE, buffer->m_delta_multiple_scattering_texture->internal_format());
        buffer->m_scattering_array[READ]->bind_image(1, 0, 0, GL_READ_WRITE, buffer->m_scattering_array[READ]->internal_format());
        buffer->m_scattering_array[WRITE]->bind_image(2, 0, 0, GL_READ_WRITE, buffer->m_scattering_array[WRITE]->internal_format());

        if (m_multiple_scattering_program->set_uniform("transmittance", 3))
            buffer->m_transmittance_array[READ]->bind(3);

        if (m_multiple_scattering_program->set_uniform("delta_scattering_density", 4))
            buffer->m_delta_scattering_density_texture->bind(4);

        m_multiple_scattering_program->set_uniform("blend", glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

        for (int layer = 0; layer < CONSTANTS::SCATTERING_DEPTH; ++layer)
        {
            m_multiple_scattering_program->set_uniform("layer", layer);
            GL_CHECK_ERROR(glDispatchCompute(CONSTANTS::SCATTERING_WIDTH / NUM_THREADS, CONSTANTS::SCATTERING_HEIGHT / NUM_THREADS, 1));
            GL_CHECK_ERROR(glFinish());
        }

        swap(buffer->m_scattering_array);
    }
}*/

}
