#include <vsg/all.h>

#ifdef vsgXchange_FOUND
#    include <vsgXchange/all.h>
#endif

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include "atmosphere.h"

vsg::ref_ptr<atmosphere::AtmosphereModel> createAtmosphereModel(vsg::ref_ptr<vsg::Window> window, vsg::ref_ptr<vsg::Options> options)
{
    // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
    // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
    // summed and averaged in each bin (e.g. the value for 360nm is the average
    // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
    // Values in W.m^-2.
    constexpr int kLambdaMin = 360;
    constexpr int kLambdaMax = 830;
    constexpr double kSolarIrradiance[48] = {
      1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
      1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
      1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
      1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
      1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
      1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
    };
    // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
    // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
    // each bin (e.g. the value for 360nm is the average of the original values
    // for all wavelengths between 360 and 370nm). Values in m^2.
    constexpr double kOzoneCrossSection[48] = {
      1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
      8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
      1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
      4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
      2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
      6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
      2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
    };
    // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
    constexpr double kDobsonUnit = 2.687e20;
    // Maximum number density of ozone molecules, in m^-3 (computed so at to get
    // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
    // the ozone density profile defined below, which is equal to 15km).
    constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
    // Wavelength independent solar irradiance "spectrum" (not physically
    // realistic, but was used in the original implementation).
    constexpr double kConstantSolarIrradiance = 1.5;
    constexpr double kBottomRadius = 6360000.0;
    constexpr double kTopRadius = 6420000.0;
    constexpr double kRayleigh = 1.24062e-6;
    constexpr double kRayleighScaleHeight = 8000.0;
    constexpr double kMieScaleHeight = 1200.0;
    constexpr double kMieAngstromAlpha = 0.0;
    constexpr double kMieAngstromBeta = 5.328e-3;
    constexpr double kMieSingleScatteringAlbedo = 0.9;
    constexpr float kMiePhaseFunctionG = 0.8f;
    constexpr double kGroundAlbedo = 0.1;
    const double max_sun_zenith_angle = 120.0 / 180.0 * vsg::PI;

    atmosphere::DensityProfileLayer rayleigh_layer{0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0};
    atmosphere::DensityProfileLayer mie_layer{0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0};

    // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
    // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
    // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
    // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).

    std::vector<double> wavelengths;
    std::vector<double> solar_irradiance;
    std::vector<double> rayleigh_scattering;
    std::vector<double> mie_scattering;
    std::vector<double> mie_extinction;
    std::vector<double> absorption_extinction;
    std::vector<double> ground_albedo;

    bool use_constant_solar_spectrum_ = false;
    bool use_ozone = true;

    for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
       double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
       double mie =
           kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
       wavelengths.push_back(l);
       if (use_constant_solar_spectrum_) {
         solar_irradiance.push_back(kConstantSolarIrradiance);
       } else {
         solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
       }
       rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
       mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
       mie_extinction.push_back(mie);
       absorption_extinction.push_back(use_ozone ?
           kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] :
           0.0);
       ground_albedo.push_back(kGroundAlbedo);
     }

    auto model = atmosphere::AtmosphereModel::create(window->getOrCreateDevice(), window->getOrCreatePhysicalDevice(), options);
    //model->compileSettings->generateDebugInfo = true;

    model->waveLengths = wavelengths;
    model->solarIrradiance = solar_irradiance;
    model->sunAngularRadius = 0.01935f;
    model->bottomRadius = 6360000.0;
    model->topRadius = 6420000.0;
    model->rayleighDensityLayer = rayleigh_layer;
    model->rayleighScattering = rayleigh_scattering;
    model->mieDensityLayer = mie_layer;
    model->mieScattering = mie_scattering;
    model->mieExtinction = mie_extinction;
    model->miePhaseFunction_g = kMiePhaseFunctionG;
    model->absorptionDensityLayer0 = atmosphere::DensityProfileLayer{static_cast<float>(25000.0 / 1000.0), 0.0f, 0.0f, static_cast<float>((1.0 / 15000.0) * 1000.0), static_cast<float>(-2.0 / 3.0)};
    model->absorptionDensityLayer1 = atmosphere::DensityProfileLayer{0.0f, 0.0f, 0.0f, static_cast<float>((-1.0 / 15000.0) * 1000.0), static_cast<float>(8.0 / 3.0)};
    model->absorptionExtinction = absorption_extinction;
    model->groundAlbedo = ground_albedo;
    model->maxSunZenithAngle = max_sun_zenith_angle;
    model->lengthUnitInMeters = 1000.0;

    model->initialize(4);

    return model;
}

int main(int argc, char** argv)
{
    try
    {
        // set up defaults and read command line arguments to override them
        vsg::CommandLine arguments(&argc, argv);

        // set up vsg::Options to pass in filepaths and ReaderWriter's and other IO related options to use when reading and writing files.
        auto options = vsg::Options::create();
        options->sharedObjects = vsg::SharedObjects::create();
        options->fileCache = vsg::getEnv("VSG_FILE_CACHE");
        options->paths = vsg::getEnvPaths("RRS2_ROOT");

#ifdef vsgXchange_all
        // add vsgXchange's support for reading and writing 3rd party file formats
        options->add(vsgXchange::all::create());
#endif

        arguments.read(options);

        auto windowTraits = vsg::WindowTraits::create();
        windowTraits->windowTitle = "vsgviewer";
        windowTraits->debugLayer = arguments.read({"--debug", "-d"});
        windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
        windowTraits->synchronizationLayer = arguments.read("--sync");
        if (int mt = 0; arguments.read({"--memory-tracking", "--mt"}, mt)) vsg::Allocator::instance()->setMemoryTracking(mt);
        if (arguments.read("--double-buffer")) windowTraits->swapchainPreferences.imageCount = 2;
        if (arguments.read("--triple-buffer")) windowTraits->swapchainPreferences.imageCount = 3; // default
        if (arguments.read("--IMMEDIATE")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        if (arguments.read("--FIFO")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        if (arguments.read("--FIFO_RELAXED")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        if (arguments.read("--MAILBOX")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        if (arguments.read({"-t", "--test"}))
        {
            windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
            windowTraits->fullscreen = true;
        }
        if (arguments.read({"--st", "--small-test"}))
        {
            windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
            windowTraits->width = 192, windowTraits->height = 108;
            windowTraits->decoration = false;
        }
        if (arguments.read({"--fullscreen", "--fs"})) windowTraits->fullscreen = true;
        if (arguments.read({"--window", "-w"}, windowTraits->width, windowTraits->height)) { windowTraits->fullscreen = false; }
        if (arguments.read({"--no-frame", "--nf"})) windowTraits->decoration = false;
        if (arguments.read("--or")) windowTraits->overrideRedirect = true;
        if (arguments.read("--d32")) windowTraits->depthFormat = VK_FORMAT_D32_SFLOAT;
        arguments.read("--screen", windowTraits->screenNum);
        arguments.read("--display", windowTraits->display);
        arguments.read("--samples", windowTraits->samples);
        auto numFrames = arguments.value(-1, "-f");
        auto pathFilename = arguments.value(std::string(), "-p");
        auto loadLevels = arguments.value(0, "--load-levels");
        auto horizonMountainHeight = arguments.value(0.0, "--hmh");
        auto sunAngle = vsg::radians(arguments.value(0.0f, "--angle"));
        if (arguments.read("--rgb")) options->mapRGBtoRGBAHint = false;

        bool generateDebug = arguments.read({"--shader-debug-info", "--sdi"});
        if (generateDebug)
            windowTraits->deviceExtensionNames.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

        if (int log_level = 0; arguments.read("--log-level", log_level)) vsg::Logger::instance()->level = vsg::Logger::Level(log_level);

        if (arguments.errors()) return arguments.writeErrorMessages(std::cerr);

        auto vsg_scene = vsg::Group::create();

        // create the viewer and assign window(s) to it
        auto viewer = vsg::Viewer::create();
        auto window = vsg::Window::create(windowTraits);
        if (!window)
        {
            std::cout << "Could not create windows." << std::endl;
            return 1;
        }

        viewer->addWindow(window);

        // compute the bounds of the scene graph to help position camera
        vsg::ComputeBounds computeBounds;
        vsg_scene->accept(computeBounds);
        vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
        double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
        double nearFarRatio = 0.001;

        // set up the camera
        auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -radius * 3.5, 0.0), centre, vsg::dvec3(0.0, 0.0, 1.0));

        vsg::ref_ptr<vsg::ProjectionMatrix> perspective;
        auto ellipsoidModel = vsg_scene->getRefObject<vsg::EllipsoidModel>("EllipsoidModel");
        if (ellipsoidModel)
        {
            perspective = vsg::EllipsoidPerspective::create(lookAt, ellipsoidModel, 30.0, static_cast<double>(window->extent2D().width) / static_cast<double>(window->extent2D().height), nearFarRatio, horizonMountainHeight);
        }
        else
        {
            perspective = vsg::Perspective::create(30.0, static_cast<double>(window->extent2D().width) / static_cast<double>(window->extent2D().height), nearFarRatio * radius, radius * 4.5);
        }

        auto camera = vsg::Camera::create(perspective, lookAt, vsg::ViewportState::create(window->extent2D()));

        // add close handler to respond the close window button and pressing escape
        viewer->addEventHandler(vsg::CloseHandler::create(viewer));

        if (pathFilename.empty())
        {
            viewer->addEventHandler(vsg::Trackball::create(camera, ellipsoidModel));
        }
        else
        {
            auto animationPath = vsg::read_cast<vsg::AnimationPath>(pathFilename, options);
            if (!animationPath)
            {
                std::cout<<"Warning: unable to read animation path : "<<pathFilename<<std::endl;
                return 1;
            }

            auto animationPathHandler = vsg::AnimationPathHandler::create(camera, animationPath, viewer->start_point());
            animationPathHandler->printFrameStatsToConsole = true;
            viewer->addEventHandler(animationPathHandler);
        }

        // if required preload specific number of PagedLOD levels.
        if (loadLevels > 0)
        {
            vsg::LoadPagedLOD loadPagedLOD(camera, loadLevels);

            auto startTime = std::chrono::steady_clock::now();

            vsg_scene->accept(loadPagedLOD);

            auto time = std::chrono::duration<float, std::chrono::milliseconds::period>(std::chrono::steady_clock::now() - startTime).count();
            std::cout << "No. of tiles loaded " << loadPagedLOD.numTiles << " in " << time << "ms." << std::endl;
        }

        auto model = createAtmosphereModel(window, options);
        //model->compileSettings->generateDebugInfo = true;

        atmosphere::RuntimeSettings settings;
        settings.white_point_exp = {1.0, 1.0, 1.0, 10.0f};
        settings.sun_direction = vsg::vec4(vsg::normalize(vsg::vec3{0.0, std::sin(sunAngle), std::cos(sunAngle)}), 0.0f);
        settings.sun_size = vsg::vec2(std::tan(0.01935f), std::cos(0.01935f));

        auto value = vsg::Value<atmosphere::RuntimeSettings>::create(settings);

        auto compute_commandGraph = model->createCubeMapGraph(value);

        auto grahics_commandGraph = vsg::createCommandGraphForView(window, camera, vsg_scene);
        viewer->assignRecordAndSubmitTaskAndPresentation({compute_commandGraph, grahics_commandGraph});

        viewer->compile();



        // rendering main loop
        while (viewer->advanceToNextFrame() && (numFrames < 0 || (numFrames--) > 0))
        {
            // pass any events into EventHandlers assigned to the Viewer
            viewer->handleEvents();

            viewer->update();

            viewer->recordAndSubmit();

            viewer->present();
        }
    }
    catch (const vsg::Exception& ve)
    {
        for (int i = 0; i < argc; ++i) std::cerr << argv[i] << " ";
        std::cerr << "\n[Exception] - " << ve.message << " result = " << ve.result << std::endl;
        return 1;
    }

    // clean up done automatically thanks to ref_ptr<>
    return 0;
}



/*
#include "skybox.h"

vsg::ref_ptr<vsg::Node> createSkybox(const vsg::Path& filename, vsg::ref_ptr<vsg::Options> options)
{
    auto data = vsg::read_cast<vsg::Data>(filename, options);
    if (!data)
    {
        std::cout << "Error: failed to load cubemap file : " << filename << std::endl;
        return {};
    }

    auto vertexShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main", skybox_vert);
    auto fragmentShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main", skybox_frag);
    const vsg::ShaderStages shaders{vertexShader, fragmentShader};

    // set up graphics pipeline
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};

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
    depthState->depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindingsDescriptions, vertexAttributeDescriptions),
        vsg::InputAssemblyState::create(),
        rasterState,
        vsg::MultisampleState::create(),
        vsg::ColorBlendState::create(),
        depthState};

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, pushConstantRanges);
    auto pipeline = vsg::GraphicsPipeline::create(pipelineLayout, shaders, pipelineStates);
    auto bindGraphicsPipeline = vsg::BindGraphicsPipeline::create(pipeline);

    // create texture image and associated DescriptorSets and binding
    auto sampler = vsg::Sampler::create();
    sampler->maxLod = data->properties.maxNumMipmaps;

    auto texture = vsg::DescriptorImage::create(sampler, data, 0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, vsg::Descriptors{texture});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSet);

    auto root = vsg::StateGroup::create();
    root->add(bindGraphicsPipeline);
    root->add(bindDescriptorSet);

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

    auto xform = vsg::MatrixTransform::create(vsg::rotate(vsg::PI * 0.5, 1.0, 0.0, 0.0));
    xform->addChild(root);

    return xform;
}

int main(int argc, char** argv)
{
    // set up defaults and read command line arguments to override them
    auto options = vsg::Options::create();
    options->paths = vsg::getEnvPaths("VSG_FILE_PATH");

#ifdef vsgXchange_all
    // add vsgXchange's support for reading and writing 3rd party file formats
    options->add(vsgXchange::all::create());
#endif

    auto windowTraits = vsg::WindowTraits::create();
    windowTraits->windowTitle = "vsgskybox";

    // set up defaults and read command line arguments to override them
    vsg::CommandLine arguments(&argc, argv);
    arguments.read(options);
    windowTraits->debugLayer = arguments.read({"--debug", "-d"});
    windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
    if (arguments.read("--double-buffer")) windowTraits->swapchainPreferences.imageCount = 2;
    if (arguments.read("--triple-buffer")) windowTraits->swapchainPreferences.imageCount = 3; // default
    if (arguments.read("--IMMEDIATE")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    if (arguments.read("--FIFO")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (arguments.read("--FIFO_RELAXED")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    if (arguments.read("--MAILBOX")) windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    if (arguments.read({"-t", "--test"}))
    {
        windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        windowTraits->fullscreen = true;
    }
    if (arguments.read({"--st", "--small-test"}))
    {
        windowTraits->swapchainPreferences.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        windowTraits->width = 192, windowTraits->height = 108;
        windowTraits->decoration = false;
    }
    if (arguments.read({"--fullscreen", "--fs"})) windowTraits->fullscreen = true;
    if (arguments.read({"--window", "-w"}, windowTraits->width, windowTraits->height)) { windowTraits->fullscreen = false; }
    if (arguments.read({"--no-frame", "--nf"})) windowTraits->decoration = false;
    if (arguments.read("--or")) windowTraits->overrideRedirect = true;
    if (arguments.read("--d32")) windowTraits->depthFormat = VK_FORMAT_D32_SFLOAT;
    arguments.read("--screen", windowTraits->screenNum);
    arguments.read("--display", windowTraits->display);
    arguments.read("--samples", windowTraits->samples);
    auto numFrames = arguments.value(-1, "-f");
    auto pathFilename = arguments.value(std::string(), "-p");
    auto horizonMountainHeight = arguments.value(0.0, "--hmh");
    auto skyboxFilename = arguments.value(vsg::Path(), "--skybox");
    auto outputFilename = arguments.value(vsg::Path(), "-o");

    if (arguments.errors()) return arguments.writeErrorMessages(std::cerr);

    auto group = vsg::Group::create();

    if (skyboxFilename)
    {
        if (auto node = createSkybox(skyboxFilename, options))
        {
            group->addChild(node);
        }
        else
        {
            return 1;
        }
    }

    vsg::Path path;

    // read any vsg files
    for (int i = 1; i < argc; ++i)
    {
        vsg::Path filename = arguments[i];
        path = vsg::filePath(filename);

        auto object = vsg::read(filename, options);
        if (auto node = object.cast<vsg::Node>())
        {
            group->addChild(node);
        }
        else if (object)
        {
            std::cout << "Unable to view object of type " << object->className() << std::endl;
        }
        else
        {
            std::cout << "Unable to load model from file " << filename << std::endl;
        }
    }

    if (group->children.empty())
    {
        std::cout << "Please specify a 3d model file on the command line." << std::endl;
        return 1;
    }

    vsg::ref_ptr<vsg::Node> vsg_scene;
    if (group->children.size() == 1)
        vsg_scene = group->children[0];
    else
        vsg_scene = group;

    if (outputFilename)
    {
        vsg::write(vsg_scene, outputFilename);
        return 0;
    }

    // create the viewer and assign window(s) to it
    auto viewer = vsg::Viewer::create();
    auto window = vsg::Window::create(windowTraits);
    if (!window)
    {
        std::cout << "Could not create windows." << std::endl;
        return 1;
    }

    viewer->addWindow(window);

    // compute the bounds of the scene graph to help position camera
    vsg::ComputeBounds computeBounds;
    vsg_scene->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    double nearFarRatio = 0.001;

    // set up the camera
    auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -radius * 3.5, 0.0), centre, vsg::dvec3(0.0, 0.0, 1.0));

    vsg::ref_ptr<vsg::ProjectionMatrix> perspective;
    if (auto ellipsoidModel = vsg_scene->getRefObject<vsg::EllipsoidModel>("EllipsoidModel"))
    {
        perspective = vsg::EllipsoidPerspective::create(lookAt, ellipsoidModel, 30.0, static_cast<double>(window->extent2D().width) / static_cast<double>(window->extent2D().height), nearFarRatio, horizonMountainHeight);
    }
    else
    {
        perspective = vsg::Perspective::create(30.0, static_cast<double>(window->extent2D().width) / static_cast<double>(window->extent2D().height), nearFarRatio * radius, radius * 4.5);
    }

    auto camera = vsg::Camera::create(perspective, lookAt, vsg::ViewportState::create(window->extent2D()));

    // add close handler to respond the close window button and pressing escape
    viewer->addEventHandler(vsg::CloseHandler::create(viewer));
    viewer->addEventHandler(vsg::Trackball::create(camera));

    auto commandGraph = vsg::createCommandGraphForView(window, camera, vsg_scene);
    viewer->assignRecordAndSubmitTaskAndPresentation({commandGraph});

    viewer->compile();

    // rendering main loop
    while (viewer->advanceToNextFrame() && (numFrames < 0 || (numFrames--) > 0))
    {
        // pass any events into EventHandlers assigned to the Viewer
        viewer->handleEvents();

        viewer->update();

        viewer->recordAndSubmit();

        viewer->present();
    }

    // clean up done automatically thanks to ref_ptr<>
    return 0;
}

int main(int argc, char** argv)
{
    // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
    // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
    // summed and averaged in each bin (e.g. the value for 360nm is the average
    // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
    // Values in W.m^-2.
    int lambda_min = 360;
    int lambda_max = 830;

    double kSolarIrradiance[] =
    {
        1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
        1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
        1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
        1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
        1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
        1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
    };

    // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
    // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
    // each bin (e.g. the value for 360nm is the average of the original values
    // for all wavelengths between 360 and 370nm). Values in m^2.
    double kOzoneCrossSection[] =
    {
        1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
        8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
        1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
        4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
        2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
        6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
        2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
    };

    // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
    double kDobsonUnit = 2.687e20;
    // Maximum number density of ozone molecules, in m^-3 (computed so at to get
    // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
    // the ozone density profile defined below, which is equal to 15km).
    double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
    // Wavelength independent solar irradiance "spectrum" (not physically
    // realistic, but was used in the original implementation).
    double kConstantSolarIrradiance = 1.5;
    double kTopRadius = 6420000.0;
    double kRayleigh = 1.24062e-6;
    double kRayleighScaleHeight = 8000.0;
    double kMieScaleHeight = 1200.0;
    double kMieAngstromAlpha = 0.0;
    double kMieAngstromBeta = 5.328e-3;
    double kMieSingleScatteringAlbedo = 0.9;
    double kMiePhaseFunctionG = 0.8;
    double kGroundAlbedo = 0.1;
    double max_sun_zenith_angle = 120.0 / 180.0 * M_PI;

    atmosphere::DensityProfileLayer rayleigh_layer{0.0f, 1.0f, static_cast<float>((-1.0 / kRayleighScaleHeight) * 1000.0), 0.0, 0.0};
    atmosphere::DensityProfileLayer mie_layer{0.0f, 1.0f, static_cast<float>((-1.0 / kMieScaleHeight) * 1000.0), 0.0, 0.0};

    // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
    // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
    // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
    // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).

    std::vector<double> wavelengths;
    std::vector<double> solar_irradiance;
    std::vector<double> rayleigh_scattering;
    std::vector<double> mie_scattering;
    std::vector<double> mie_extinction;
    std::vector<double> absorption_extinction;
    std::vector<double> ground_albedo;

    for (int l = lambda_min; l <= lambda_max; l += 10)
    {
        double lambda = l * 1e-3;  // micro-meters
        double mie = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);

        wavelengths.push_back(l);

        solar_irradiance.push_back(kSolarIrradiance[(l - lambda_min) / 10]);

        rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
        mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
        mie_extinction.push_back(mie);
        absorption_extinction.push_back(kMaxOzoneNumberDensity * kOzoneCrossSection[(l - lambda_min) / 10]);
        ground_albedo.push_back(kGroundAlbedo);
    }

    auto options = vsg::Options::create();
    options->paths = vsg::getEnvPaths("RRS2_ROOT");

    auto model = atmosphere::AtmosphereModel::create(options);
    model->compileSettings->generateDebugInfo = true;

    model->waveLengths = wavelengths;
    model->solarIrradiance = solar_irradiance;
    model->sunAngularRadius = 0.01935f;
    model->bottomRadius = 6360000.0;
    model->topRadius = 6420000.0;
    model->rayleighDensityLayer = rayleigh_layer;
    model->rayleighScattering = rayleigh_scattering;
    model->mieDensityLayer = mie_layer;
    model->mieScattering = mie_scattering;
    model->mieExtinction = mie_extinction;
    model->miePhaseFunction_g = 0.8f;
    model->absorptionDensityLayer0 = atmosphere::DensityProfileLayer{static_cast<float>(25000.0 / 1000.0), 0.0f, 0.0f, static_cast<float>((1.0 / 15000.0) * 1000.0), static_cast<float>(-2.0 / 3.0)};
    model->absorptionDensityLayer1 = atmosphere::DensityProfileLayer{0.0f, 0.0f, 0.0f, static_cast<float>((-1.0 / 15000.0) * 1000.0), static_cast<float>(8.0 / 3.0)};
    model->absorptionExtinction = absorption_extinction;
    model->groundAlbedo = ground_albedo;
    model->maxSunZenithAngle = max_sun_zenith_angle;
    model->lengthUnitInMeters = 1000.0f;

    int num_scattering_orders = 4;

    model->initialize(num_scattering_orders);

}
*/
