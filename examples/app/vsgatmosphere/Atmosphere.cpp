#include "Atmosphere.h"
#include "AtmosphereTools.h"

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
    generateTextures();

    assignComputeConstants();
    assignRenderConstants();

    auto memoryBarrier = vsg::MemoryBarrier::create();
    memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    auto memoryBarrierCmd = vsg::PipelineBarrier::create(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, memoryBarrier);

    auto singlePass = vsg::Commands::create();
    auto transmittanceCommands = vsg::Commands::create();
    auto multipleScattering = vsg::Commands::create();

    auto lfr = vsg::mat4Value::create(vsg::mat4());
    auto parameters = vsg::Value<Parameters>::create(Parameters());
    auto profiles = vsg::Array<DensityProfileLayer>::create(4);
    profiles->set(0, rayleighDensityLayer);
    profiles->set(1, mieDensityLayer);
    profiles->set(2, absorptionDensityLayer0);
    profiles->set(3, absorptionDensityLayer1);

    auto pLayout = parametersLayout();

    auto lfrBuffer = vsg::DescriptorBuffer::create(lfr, 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto parametersBuffer = vsg::DescriptorBuffer::create(parameters, 1, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto proflesArray = vsg::DescriptorBuffer::create(profiles, 2, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    vsg::ref_ptr<vsg::BindDescriptorSet> bindParameters;
    vsg::ref_ptr<vsg::BindDescriptorSet> bindOrder;

    {
        vsg::Descriptors descriptors{lfrBuffer, parametersBuffer, proflesArray};
        auto descriptorSet = vsg::DescriptorSet::create(pLayout, descriptors);
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{pLayout}, vsg::PushConstantRanges{});
        bindParameters = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, descriptorSet);
        bindParameters->slot = 2;
    }

    auto oLayout = orderLayout();

    auto order = vsg::intValue::create(0);
    auto orderBuffer = vsg::DescriptorBuffer::create(order, 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    {
        vsg::Descriptors descriptors{orderBuffer};
        auto descriptorSet = vsg::DescriptorSet::create(oLayout, descriptors);
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{oLayout}, vsg::PushConstantRanges{});
        bindOrder = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 2, descriptorSet);
        bindOrder->slot = 3;
    }

    {
        auto texturesSet = bindTransmittance();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_transmittance_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        transmittanceCommands->addChild(bindPipeline);
        transmittanceCommands->addChild(bindTextures);
        transmittanceCommands->addChild(vsg::Dispatch::create(uint32_t(ceil(float(transmittanceWidth) / float(numThreads))),
                                                     uint32_t(ceil(float(transmittanceHeight) / float(numThreads))), 1));
        transmittanceCommands->addChild(memoryBarrierCmd);
        singlePass->addChild(transmittanceCommands);
    }

    {
        auto texturesSet = bindDirectIrradiance();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout}, vsg::PushConstantRanges{});
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
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_single_scattering_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        singlePass->addChild(bindPipeline);
        singlePass->addChild(bindTextures);
        singlePass->addChild(vsg::Dispatch::create(uint32_t(scatteringWidth() / numThreads),
                                                     uint32_t(scatteringHeight() / numThreads),
                                                     uint32_t(scatteringDepth() / numThreads)));
        singlePass->addChild(memoryBarrierCmd);
    }

    {
        auto texturesSet = bindScatteringDensity();
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout, oLayout}, vsg::PushConstantRanges{});
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
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout, oLayout}, vsg::PushConstantRanges{});
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
        auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{texturesSet->setLayout, pLayout, oLayout}, vsg::PushConstantRanges{});
        auto bindPipeline = bindCompute("shaders/scattering/compute_multiple_scattering_cs.glsl", pipelineLayout);
        auto bindTextures = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, texturesSet);
        multipleScattering->addChild(bindPipeline);
        multipleScattering->addChild(bindTextures);
        multipleScattering->addChild(vsg::Dispatch::create(uint32_t(ceil(float(scatteringWidth()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringHeight()) / float(numThreads))),
                                                     uint32_t(ceil(float(scatteringDepth()) / float(numThreads)))));
        multipleScattering->addChild(memoryBarrierCmd);
    }

    // compile the Vulkan objects
    auto compileTraversal = vsg::CompileTraversal::create(_device);
    auto context = compileTraversal->contexts.front();

    bindParameters->accept(*compileTraversal);
    bindOrder->accept(*compileTraversal);
    singlePass->accept(*compileTraversal);
    multipleScattering->accept(*compileTraversal);

    int computeQueueFamily = _physicalDevice->getQueueFamily(VK_QUEUE_COMPUTE_BIT);

    context->commandPool = vsg::CommandPool::create(_device, computeQueueFamily);

    auto fence = vsg::Fence::create(_device);
    auto computeQueue = _device->getQueue(computeQueueFamily);

    int num_iterations = (precomputedWavelenghts + 2) / 3;
    double dlambda = (kLambdaMax - kLambdaMin) / (3 * num_iterations);
    for (int i = 0; i < num_iterations; ++i)
    {
        vsg::vec3 lambdas(
                kLambdaMin + (3 * i + 0.5) * dlambda,
                kLambdaMin + (3 * i + 1.5) * dlambda,
                kLambdaMin + (3 * i + 2.5) * dlambda);

        auto coeff = [dlambda](double lambda, int component)
        {
            // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid
            // artefacts due to too large values when using half precision on GPU.
            // We add this term back in kAtmosphereShader, via
            // SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
            // Model constructor).
            double x = cieColorMatchingFunctionTableValue(lambda, 1);
            double y = cieColorMatchingFunctionTableValue(lambda, 2);
            double z = cieColorMatchingFunctionTableValue(lambda, 3);
            return static_cast<float>((
                XYZ_TO_SRGB[component * 3] * x +
                XYZ_TO_SRGB[component * 3 + 1] * y +
                XYZ_TO_SRGB[component * 3 + 2] * z) * dlambda);
        };
        vsg::mat4 luminance_from_radiance{
                  coeff(lambdas[0], 0), coeff(lambdas[0], 1), coeff(lambdas[0], 2), 0.0f,
                  coeff(lambdas[1], 0), coeff(lambdas[1], 1), coeff(lambdas[1], 2), 0.0f,
                  coeff(lambdas[2], 0), coeff(lambdas[2], 1), coeff(lambdas[2], 2), 0.0f,
                  0.0, 0.0f, 0.0f, 1.0f};

        lfr->set(luminance_from_radiance);
        computeParameters(parameters, lambdas);

        lfrBuffer->copyDataListToBuffers();
        parametersBuffer->copyDataListToBuffers();

        vsg::submitCommandsToQueue(context->commandPool, fence, 100000000000, computeQueue, [&](vsg::CommandBuffer& commandBuffer) {
            bindParameters->record(commandBuffer);
            singlePass->record(commandBuffer);
        });

        // Compute the 2nd, 3rd and 4th orderValue of scattering, in sequence.
        for (int o = 2; o <= scatteringOrders; ++o)
        {
            order->set(o);
            orderBuffer->copyDataListToBuffers();

            vsg::submitCommandsToQueue(context->commandPool, fence, 100000000000, computeQueue, [&](vsg::CommandBuffer& commandBuffer) {
                bindParameters->record(commandBuffer);
                bindOrder->record(commandBuffer);
                multipleScattering->record(commandBuffer);
            });
        }
    }

    lfr->set(vsg::mat4());
    computeParameters(parameters, {kLambdaR, kLambdaG, kLambdaB});

    lfrBuffer->copyDataListToBuffers();
    parametersBuffer->copyDataListToBuffers();

    vsg::submitCommandsToQueue(context->commandPool, fence, 100000000000, computeQueue, [&](vsg::CommandBuffer& commandBuffer) {
        bindParameters->record(commandBuffer);
        transmittanceCommands->record(commandBuffer);
    });
}

vsg::vec3 AtmosphereModel::convertSpectrumToLinearSrgb(double c)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    const int dlambda = 1;
    for (int lambda = kLambdaMin; lambda < kLambdaMax; lambda += dlambda) {
      double value = interpolate(waveLengths, solarIrradiance, lambda);
      x += cieColorMatchingFunctionTableValue(lambda, 1) * value;
      y += cieColorMatchingFunctionTableValue(lambda, 2) * value;
      z += cieColorMatchingFunctionTableValue(lambda, 3) * value;
    }
    auto r = MAX_LUMINOUS_EFFICACY *
        (XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z) * dlambda;
    auto g = MAX_LUMINOUS_EFFICACY *
        (XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z) * dlambda;
    auto b = MAX_LUMINOUS_EFFICACY *
        (XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z) * dlambda;

    double white_point = (r + g + b) / c;

    r /= white_point;
    g /= white_point;
    b /= white_point;

    return {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b)};
}

vsg::ref_ptr<vsg::CommandGraph> AtmosphereModel::createCubeMapGraph(vsg::ref_ptr<vsg::Value<RuntimeSettings>> settings, vsg::ref_ptr<vsg::vec4Value> camera)
{
    auto computeShader = vsg::ShaderStage::read(VK_SHADER_STAGE_COMPUTE_BIT, "main", "shaders/scattering/reflection_map.glsl", _options);
    computeShader->module->hints = compileSettings;
    computeShader->specializationConstants = _renderConstants;

    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    vsg::PushConstantRanges pushConstantRanges{
        {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vsg::vec4)}
    };

    auto settingsBuffer = vsg::DescriptorBuffer::create(settings, 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto scatteringTexture = vsg::DescriptorImage::create(_scatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto singleMieTexture = vsg::DescriptorImage::create(_singleMieScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto cubemap = vsg::DescriptorImage::create(_cubeMap, 5, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, pushConstantRanges);

    vsg::Descriptors descriptors{settingsBuffer, transmittanceTexture, irradianceTexture, scatteringTexture, singleMieTexture, cubemap};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, descriptorSet);

    auto pushCamera = vsg::PushConstants::create(VK_SHADER_STAGE_COMPUTE_BIT, 0, camera);

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
    compute_commandGraph->addChild(pushCamera);
    auto workgroups = uint32_t(ceil(float(cubeSize) / float(numViewerThreads)));
    compute_commandGraph->addChild(vsg::Dispatch::create(workgroups, workgroups, 6));
    compute_commandGraph->addChild(postCopyBarrierCmd);

    return compute_commandGraph;
}

vsg::ref_ptr<vsg::Node> AtmosphereModel::createSkyBox()
{
    auto vertexShader = vsg::ShaderStage::read(VK_SHADER_STAGE_VERTEX_BIT, "main", "shaders/scattering/cubemap_vs.glsl", _options);
    auto fragmentShader = vsg::ShaderStage::read(VK_SHADER_STAGE_FRAGMENT_BIT, "main", "shaders/scattering/cubemap_fs.glsl", _options);
    const vsg::ShaderStages shaders{vertexShader, fragmentShader};

    // set up graphics pipeline
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);


    vsg::PushConstantRanges pushConstantRanges{
        {VK_SHADER_STAGE_VERTEX_BIT, 0, 128} // projection view, and model matrices, actual push constant calls automatically provided by the VSG's DispatchTraversal
    };

    vsg::VertexInputState::Bindings vertexBindingsDescriptions{
        VkVertexInputBindingDescription{0, sizeof(vsg::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};

    vsg::VertexInputState::Attributes vertexAttributeDescriptions{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};

    auto rasterState = vsg::RasterizationState::create();
    rasterState->cullMode = VK_CULL_MODE_FRONT_BIT;

    auto depthState = vsg::DepthStencilState::create();
    depthState->depthTestEnable = VK_TRUE;
    depthState->depthWriteEnable = VK_FALSE;
    depthState->depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    auto colorBlendState = vsg::ColorBlendState::create();
    colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {true, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_SUBTRACT, VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT}};


    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindingsDescriptions, vertexAttributeDescriptions),
        vsg::InputAssemblyState::create(),
        rasterState,
        vsg::MultisampleState::create(),
        colorBlendState,
        depthState};

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, pushConstantRanges);
    auto pipeline = vsg::GraphicsPipeline::create(pipelineLayout, shaders, pipelineStates);
    auto bindGraphicsPipeline = vsg::BindGraphicsPipeline::create(pipeline);

    auto root = vsg::Commands::create();
    root->addChild(bindGraphicsPipeline);

    auto texture = vsg::DescriptorImage::create(_cubeMap, 0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, vsg::Descriptors{texture});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSet);
    root->addChild(bindDescriptorSet);

    auto vertices = vsg::vec3Array::create({// Back
                                            {-1.0f, -1.0f, -1.0f},
                                            {1.0f, -1.0f, -1.0f},
                                            {-1.0f, 1.0f, -1.0f},
                                            {1.0f, 1.0f, -1.0f},

                                            // Front
                                            {-1.0f, -1.0f, 1.0f},
                                            {1.0f, -1.0f, 1.0f},
                                            {-1.0f, 1.0f, 1.0f},
                                            {1.0f, 1.0f, 1.0f},

                                            // Left
                                            {-1.0f, -1.0f, -1.0f},
                                            {-1.0f, -1.0f, 1.0f},
                                            {-1.0f, 1.0f, -1.0f},
                                            {-1.0f, 1.0f, 1.0f},

                                            // Right
                                            {1.0f, -1.0f, -1.0f},
                                            {1.0f, -1.0f, 1.0f},
                                            {1.0f, 1.0f, -1.0f},
                                            {1.0f, 1.0f, 1.0f},

                                            // Bottom
                                            {-1.0f, -1.0f, -1.0f},
                                            {-1.0f, -1.0f, 1.0f},
                                            {1.0f, -1.0f, -1.0f},
                                            {1.0f, -1.0f, 1.0f},

                                            // Top
                                            {-1.0f, 1.0f, -1.0f},
                                            {-1.0f, 1.0f, 1.0f},
                                            {1.0f, 1.0f, -1.0f},
                                            {1.0f, 1.0f, 1.0}});

    auto indices = vsg::ushortArray::create({// Back
                                             0, 2, 1,
                                             1, 2, 3,

                                             // Front
                                             6, 4, 5,
                                             7, 6, 5,

                                             // Left
                                             10, 8, 9,
                                             11, 10, 9,

                                             // Right
                                             14, 13, 12,
                                             15, 13, 14,

                                             // Bottom
                                             17, 16, 19,
                                             19, 16, 18,

                                             // Top
                                             23, 20, 21,
                                             22, 20, 23});

    root->addChild(vsg::BindVertexBuffers::create(0, vsg::DataList{vertices}));
    root->addChild(vsg::BindIndexBuffer::create(indices));
    root->addChild(vsg::DrawIndexed::create(indices->size(), 1, 0, 0, 0));

    auto xform = vsg::MatrixTransform::create(vsg::rotate(vsg::PI, 0.0, 1.0, 0.0));
    xform->addChild(root);

    return xform;
}

vsg::ref_ptr<vsg::Node> AtmosphereModel::createSky(vsg::ref_ptr<vsg::Value<RuntimeSettings>> settings, vsg::ref_ptr<vsg::mat4Array> inverseMatrices)
{
    auto vertexShader = vsg::ShaderStage::read(VK_SHADER_STAGE_VERTEX_BIT, "main", "shaders/scattering/sky_vs.glsl", _options);
    auto fragmentShader = vsg::ShaderStage::read(VK_SHADER_STAGE_FRAGMENT_BIT, "main", "shaders/scattering/sky_fs.glsl", _options);
    const vsg::ShaderStages shaders{vertexShader, fragmentShader};

    fragmentShader->specializationConstants = _renderConstants;
    vertexShader->module->hints = compileSettings;
    fragmentShader->module->hints = compileSettings;

    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    vsg::PushConstantRanges pushConstantRanges{
        {VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, 0, 256}
    };

    vsg::VertexInputState::Bindings vertexBindingsDescriptions{
        VkVertexInputBindingDescription{0, sizeof(vsg::vec2), VK_VERTEX_INPUT_RATE_VERTEX}};

    vsg::VertexInputState::Attributes vertexAttributeDescriptions{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};

    auto depthState = vsg::DepthStencilState::create();
    depthState->depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    auto colorBlendState = vsg::ColorBlendState::create();
    colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {true, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_SUBTRACT, VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT}};


    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindingsDescriptions, vertexAttributeDescriptions),
        vsg::InputAssemblyState::create(),
        vsg::RasterizationState::create(),
        vsg::MultisampleState::create(),
        colorBlendState,
        depthState};

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, pushConstantRanges);
    auto pipeline = vsg::GraphicsPipeline::create(pipelineLayout, shaders, pipelineStates);
    auto bindGraphicsPipeline = vsg::BindGraphicsPipeline::create(pipeline);

    auto root = vsg::StateGroup::create();
    root->add(bindGraphicsPipeline);

    auto settingsBuffer = vsg::DescriptorBuffer::create(settings, 0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    auto transmittanceTexture = vsg::DescriptorImage::create(_transmittanceTexture, 1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto scatteringTexture = vsg::DescriptorImage::create(_scatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto singleMieTexture = vsg::DescriptorImage::create(_singleMieScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    vsg::Descriptors descriptors{settingsBuffer, transmittanceTexture, irradianceTexture, scatteringTexture, singleMieTexture};
    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, descriptors);
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, descriptorSet);
    root->add(bindDescriptorSet);

    auto pushInverse = vsg::PushConstants::create(VK_SHADER_STAGE_VERTEX_BIT, 128, inverseMatrices);
    root->add(pushInverse);

    auto vid = vsg::VertexIndexDraw::create();

    auto vertices = vsg::vec2Array::create({
                                            {-1.0f, -1.0f},
                                            {1.0f, -1.0f},
                                            {-1.0f, 1.0f},
                                            {1.0f, 1.0f}});

    auto indices = vsg::ushortArray::create({0, 2, 1, 1, 2, 3});

    vid->assignArrays(vsg::DataList{vertices});
    vid->assignIndices(indices);
    vid->indexCount = static_cast<uint32_t>(indices->size());
    vid->instanceCount = 1;

    root->addChild(vid);

    return root;
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
    computeStage->specializationConstants = _computeConstants;

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
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    auto deltaIrradianceTexture = vsg::DescriptorImage::create(_deltaIrradianceTexture, 0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    auto irradianceTexture = vsg::DescriptorImage::create(_irradianceTexture, 1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    auto singleTexture = vsg::DescriptorImage::create(_deltaRayleighScatteringTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto mieTexture = vsg::DescriptorImage::create(_deltaMieScatteringTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    auto multipleTexture = vsg::DescriptorImage::create(_deltaMultipleScatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

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

void AtmosphereModel::computeParameters(vsg::ref_ptr<vsg::Value<Parameters>> parameters, const vsg::vec3 &lambdas) const
{
    Parameters p;
    p.solar_irradiance = toVector(waveLengths, solarIrradiance, lambdas, 1.0);
    p.rayleigh_scattering = toVector(waveLengths, rayleighScattering, lambdas, lengthUnitInMeters);
    p.mie_scattering = toVector(waveLengths, mieScattering, lambdas, lengthUnitInMeters);
    p.mie_extinction = toVector(waveLengths, mieExtinction, lambdas, lengthUnitInMeters);
    p.absorption_extinction = toVector(waveLengths, absorptionExtinction, lambdas, lengthUnitInMeters);
    p.ground_albedo = toVector(waveLengths, groundAlbedo, lambdas, 1.0);

    parameters->set(p);
}

vsg::ref_ptr<vsg::DescriptorSetLayout> AtmosphereModel::parametersLayout() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    return vsg::DescriptorSetLayout::create(descriptorBindings);
}

vsg::ref_ptr<vsg::DescriptorSetLayout> AtmosphereModel::orderLayout() const
{
    // set up DescriptorSetLayout, DecriptorSet and BindDescriptorSets
    vsg::DescriptorSetLayoutBindings descriptorBindings{{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    return vsg::DescriptorSetLayout::create(descriptorBindings);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AtmosphereModel::assignComputeConstants()
{
    _computeConstants = {
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

        {36, vsg::floatValue::create(static_cast<float>(sunAngularRadius))},
        {37, vsg::floatValue::create(static_cast<float>(bottomRadius / lengthUnitInMeters))},
        {38, vsg::floatValue::create(static_cast<float>(topRadius / lengthUnitInMeters))},
        {39, vsg::floatValue::create(static_cast<float>(miePhaseFunction_g))},
        {40, vsg::floatValue::create(static_cast<float>(std::cos(maxSunZenithAngle)))}
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

    _renderConstants = {
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
        {46, vsg::floatValue::create(SUN_SPECTRAL_RADIANCE_TO_LUMINANCE.z)}
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

}
