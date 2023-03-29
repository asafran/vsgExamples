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

    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(transmittanceCommands());
    commandGraph->addChild(directIrradianceCommands());
    commandGraph->addChild(singleScatteringCommands());

    // Compute the 2nd, 3rd and 4th orderValue of scattering, in sequence.
    for (int order = 2; order <= scatteringOrders; ++order)
    {
        auto orderValue = vsg::intValue::create(order);

        commandGraph->addChild(scatteringDensityCommands(orderValue));
        commandGraph->addChild(indirectIrradianceCommands(orderValue));
        commandGraph->addChild(multipleScatteringCommands(orderValue));
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

vsg::ref_ptr<vsg::CommandGraph> AtmosphereModel::createCubeMapGraph(vsg::ref_ptr<vsg::Value<RuntimeSettings> > settings) const
{
    auto computeShader = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/reflection_map.glsl", _options);
    computeShader->module->hints = compileSettings;
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

    _irradianceTexture = generate2D(irradianceWidth, irradianceHeight);
    _deltaIrradianceTexture = generate2D(irradianceWidth, irradianceHeight);

    auto width = scatteringWidth();
    auto height = scatteringHeight();
    auto depth = scatteringDepth();

    _deltaRayleighScatteringTexture = generate3D(width, height, depth);
    _deltaMieScatteringTexture = generate3D(width, height, depth);
    _scatteringTexture = generate3D(width, height, depth);
    _singleMieScatteringTexture = generate3D(width, height, depth);

    _deltaScatteringDensityTexture = generate3D(width, height, depth);
    _deltaMultipleScatteringTexture = generate3D(width, height, depth);

    _cubeMap = generateCubemap(cubeSize);
}

vsg::ref_ptr<vsg::ImageInfo> AtmosphereModel::generate2D(uint32_t width, uint32_t height)
{
    vsg::ref_ptr<vsg::Image> image = vsg::Image::create();
    image->usage |= (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    image->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image->mipLevels = 1;
    image->extent = VkExtent3D{width, height, 1};
    image->imageType = VK_IMAGE_TYPE_2D;
    image->arrayLayers = 1;

    image->compile(_device);
    image->allocateAndBindMemory(_device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    auto imageView = vsg::ImageView::create(image, VK_IMAGE_ASPECT_COLOR_BIT);
    auto sampler = vsg::Sampler::create();
    return vsg::ImageInfo::create(sampler, imageView, VK_IMAGE_LAYOUT_GENERAL);

}

vsg::ref_ptr<vsg::ImageInfo> AtmosphereModel::generate3D(uint32_t width, uint32_t height, uint32_t depth)
{
    vsg::ref_ptr<vsg::Image> image = vsg::Image::create();
    image->usage |= (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    image->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image->mipLevels = 1;
    image->extent = VkExtent3D{width, height, depth};
    image->imageType = VK_IMAGE_TYPE_3D;
    image->arrayLayers = 1;

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

vsg::ref_ptr<vsg::BindComputePipeline> AtmosphereModel::bindCompute(const vsg::Path& filename, vsg::ref_ptr<vsg::PipelineLayout> layout)
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", filename, _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(layout, computeStage);

    return vsg::BindComputePipeline::create(pipeline);;
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

vsg::ref_ptr<vsg::Commands> AtmosphereModel::directIrradianceCommands() const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_direct_irradiance_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
         {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
         {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto writeTexture = vsg::DescriptorImage::create(_irradianceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto deltaTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{writeTexture, deltaTexture, transmittanceTexture};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(ceil(float(irradianceWidth) / float(numThreads))),
                                                  uint32_t(ceil(float(irradianceHeight) / float(numThreads))), 1));
    commandGraph->addChild(memoryBarrierCmd);

    return commandGraph;
}

vsg::ref_ptr<vsg::Commands> AtmosphereModel::singleScatteringCommands() const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_single_scattering_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

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
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(scatteringWidth() / numThreads),
                                                 uint32_t(scatteringHeight() / numThreads),
                                                 uint32_t(scatteringDepth() / numThreads)));
    commandGraph->addChild(memoryBarrierCmd);

    return commandGraph;
}

vsg::ref_ptr<vsg::Commands> AtmosphereModel::multipleScatteringCommands(vsg::ref_ptr<vsg::intValue> orderValue) const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_multiple_scattering_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto deltaMultiple = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto scattering = vsg::DescriptorImage::create(_scatteringTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto singleTexture = vsg::DescriptorImage::create(_transmittanceTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto densityTexture = vsg::DescriptorImage::create(_deltaScatteringDensityTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto orderValueBuffer = vsg::DescriptorBuffer::create(orderValue, 5);

    vsg::Descriptors descriptors{deltaMultiple, scattering, singleTexture, densityTexture, orderValueBuffer};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(ceil(float(scatteringWidth()) / float(numThreads))),
                                                 uint32_t(ceil(float(scatteringHeight()) / float(numThreads))),
                                                 uint32_t(ceil(float(scatteringDepth()) / float(numThreads)))));
    commandGraph->addChild(memoryBarrierCmd);
    return commandGraph;
}

vsg::ref_ptr<vsg::Commands> AtmosphereModel::indirectIrradianceCommands(vsg::ref_ptr<vsg::intValue> orderValue) const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_indirect_irradiance_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;

    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto deltaIrradianceTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto singleTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto multipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto orderValueBuffer = vsg::DescriptorBuffer::create(orderValue, 6);

    vsg::Descriptors descriptors{deltaIrradianceTexture, irradianceTexture, singleTexture,
                mieTexture, multipleTexture, orderValueBuffer};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(ceil(float(irradianceWidth) / float(numThreads))),
                                                 uint32_t(ceil(float(irradianceHeight) / float(numThreads))), 1));
    commandGraph->addChild(memoryBarrierCmd);
    return commandGraph;
}

vsg::ref_ptr<vsg::Commands> AtmosphereModel::scatteringDensityCommands(vsg::ref_ptr<vsg::intValue> orderValue) const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_scattering_density_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto scatteringTexture = vsg::DescriptorImage::create(_deltaScatteringDensityTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto rayTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto multipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto irradianceTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto orderValueBuffer = vsg::DescriptorBuffer::create(orderValue, 6);

    vsg::Descriptors descriptors{scatteringTexture, transmittanceTexture, rayTexture, mieTexture, multipleTexture, irradianceTexture, orderValueBuffer};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(ceil(float(scatteringWidth()) / float(numThreads))),
                                                 uint32_t(ceil(float(scatteringHeight()) / float(numThreads))),
                                                 uint32_t(ceil(float(scatteringDepth()) / float(numThreads)))));
    commandGraph->addChild(memoryBarrierCmd);
    return commandGraph;
}

vsg::ref_ptr<vsg::Commands> AtmosphereModel::transmittanceCommands() const
{
    auto computeStage = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/compute_transmittance_cs.glsl", _options);
    computeStage->module->hints = compileSettings;
    computeStage->specializationConstants = _constants;
    /*
    computeStage->specializationConstants = vsg::ShaderStage::SpecializationConstants{
        {0, vsg::intValue::create(transmittanceWidth)},
        {1, vsg::intValue::create(transmittanceHeight)},
        {2, vsg::intValue::create(NUM_THREADS)}};
    */
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto writeTexture = vsg::DescriptorImage::create(_transmittanceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    vsg::Descriptors descriptors{writeTexture};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, vsg::PushConstantRanges{});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    // set up the compute pipeline
    auto pipeline = vsg::ComputePipeline::create(pipelineLayout, computeStage);
    auto bindPipeline = vsg::BindComputePipeline::create(pipeline);

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    // assign to a Commands that binds the Pipeline and DescritorSets and calls Dispatch
    auto commandGraph = vsg::Commands::create();
    commandGraph->addChild(bindPipeline);
    commandGraph->addChild(bindDescriptorSet);
    commandGraph->addChild(vsg::Dispatch::create(uint32_t(ceil(float(transmittanceWidth) / float(numThreads))),
                                                 uint32_t(ceil(float(transmittanceHeight) / float(numThreads))), 1));
    commandGraph->addChild(memoryBarrierCmd);

    return commandGraph;
/*
    // compile the Vulkan objects
    auto compileTraversal = vsg::CompileTraversal::create(_device);
    auto context = compileTraversal->contexts.front();
    context->commandPool = vsg::CommandPool::create(_device, _computeQueueFamily);
    // context->descriptorPool = vsg::DescriptorPool::create(device, 1, vsg::DescriptorPoolSizes{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}});

    commandGraph->accept(*compileTraversal);

    // setup fence
    auto fence = vsg::Fence::create(_device);

    // submit commands
    vsg::submitCommandsToQueue(context->commandPool, fence, 100000000000, _queue, [&](vsg::CommandBuffer& commandBuffer) {
        commandGraph->record(commandBuffer);
    });

    // Map the buffer memory and assign as a vec4Array2D that will automatically unmap itself on destruction.
    auto image = vsg::MappedData<vsg::vec4Array2D>::create(bufferMemory, 0, 0, vsg::Data::Properties{VK_FORMAT_R32G32B32A32_SFLOAT}, width, height); // deviceMemory, offset, flags an d dimensions
    return image;
*/
}


void AtmosphereModel::setupShaderConstants()
{
    //auto SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = compute_spectral_radiance_to_luminance_factors(waveLengths, solarIrradiance, -3);
    //auto SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = compute_spectral_radiance_to_luminance_factors(waveLengths, solarIrradiance, 0);

    auto solar_irradiance = to_vector(waveLengths, solarIrradiance, 1.0);

    auto rayleigh_scattering = to_vector(waveLengths, rayleighScattering, lengthUnitInMeters);

    auto mie_scattering = to_vector(waveLengths, mieScattering, lengthUnitInMeters);
    auto mie_extinction = to_vector(waveLengths, mieExtinction, lengthUnitInMeters);

    auto absorption_extinction = to_vector(waveLengths, absorptionExtinction, lengthUnitInMeters);

    auto ground_albedo = to_vector(waveLengths, groundAlbedo, 1.0);

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

        {36, vsg::floatValue::create(sunAngularRadius)},
        {37, vsg::floatValue::create(bottomRadius)},
        {38, vsg::floatValue::create(topRadius )},
        {39, vsg::floatValue::create(miePhaseFunction_g)},
        {40, vsg::floatValue::create(static_cast<float>(std::cos(maxSunZenithAngle)))},
/*
        {41, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {42, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {43, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},

        {44, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {45, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {46, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},
*/
        {47, vsg::floatValue::create(rayleighDensityLayer.width)},
        {48, vsg::floatValue::create(rayleighDensityLayer.exp_term)},
        {49, vsg::floatValue::create(rayleighDensityLayer.exp_scale)},
        {50, vsg::floatValue::create(rayleighDensityLayer.linear_term)},
        {51, vsg::floatValue::create(rayleighDensityLayer.constant_term)},

        {52, vsg::floatValue::create(mieDensityLayer.width)},
        {53, vsg::floatValue::create(mieDensityLayer.exp_term)},
        {54, vsg::floatValue::create(mieDensityLayer.exp_scale)},
        {55, vsg::floatValue::create(mieDensityLayer.linear_term)},
        {56, vsg::floatValue::create(mieDensityLayer.constant_term)},

        {57, vsg::floatValue::create(absorptionDensityLayer0.width)},
        {58, vsg::floatValue::create(absorptionDensityLayer0.exp_term)},
        {59, vsg::floatValue::create(absorptionDensityLayer0.exp_scale)},
        {60, vsg::floatValue::create(absorptionDensityLayer0.linear_term)},
        {61, vsg::floatValue::create(absorptionDensityLayer0.constant_term)},

        {62, vsg::floatValue::create(absorptionDensityLayer1.width)},
        {63, vsg::floatValue::create(absorptionDensityLayer1.exp_term)},
        {64, vsg::floatValue::create(absorptionDensityLayer1.exp_scale)},
        {65, vsg::floatValue::create(absorptionDensityLayer1.linear_term)},
        {66, vsg::floatValue::create(absorptionDensityLayer1.constant_term)}
    };
}

void AtmosphereModel::assignRenderConstants()
{
    //auto SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = compute_spectral_radiance_to_luminance_factors(waveLengths, solarIrradiance, -3);
    //auto SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = compute_spectral_radiance_to_luminance_factors(waveLengths, solarIrradiance, 0);

    auto solar_irradiance = to_vector(waveLengths, solarIrradiance, 1.0);

    auto rayleigh_scattering = to_vector(waveLengths, rayleighScattering, lengthUnitInMeters);
    auto ground_albedo = to_vector(waveLengths, groundAlbedo, 1.0);

    auto mie_scattering = to_vector(waveLengths, mieScattering, lengthUnitInMeters);

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

        {27, vsg::floatValue::create(solar_irradiance.x)},
        {28, vsg::floatValue::create(solar_irradiance.y)},
        {29, vsg::floatValue::create(solar_irradiance.z)},

        {33, vsg::floatValue::create(ground_albedo.x)},
        {34, vsg::floatValue::create(ground_albedo.y)},
        {35, vsg::floatValue::create(ground_albedo.z)},

        {36, vsg::floatValue::create(sunAngularRadius)},
        {37, vsg::floatValue::create(bottomRadius)},
        {38, vsg::floatValue::create(topRadius)},
        {39, vsg::floatValue::create(miePhaseFunction_g)},
        {40, vsg::floatValue::create(static_cast<float>(std::cos(maxSunZenithAngle)))},
/*
        {41, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {42, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {43, vsg::floatValue::create(SKY_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},

        {44, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.x)},
        {45, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.y)},
        {46, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.z)},
*/
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
