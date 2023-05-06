#include "AtmosphereLighting.h"
#include <vsg/io/read.h>
#include <vsg/state/material.h>
#include <vsg/state/DescriptorImage.h>

namespace atmosphere {

    AtmosphereLighting::AtmosphereLighting(vsg::ref_ptr<AtmosphereModel> model, uint32_t maxNumberLights, uint32_t maxViewports)
        : vsg::Inherit<vsg::ViewDependentState, AtmosphereLighting>(maxNumberLights, maxViewports)
        , atmosphereModel(model)
    {
        descriptorSet->descriptors.push_back(vsg::DescriptorImage::create(model->_transmittanceTexture, 2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        descriptorSet->descriptors.push_back(vsg::DescriptorImage::create(model->_irradianceTexture, 3, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        descriptorSet->descriptors.push_back(vsg::DescriptorImage::create(model->_scatteringTexture, 4, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        descriptorSet->descriptors.push_back(vsg::DescriptorImage::create(model->_singleMieScatteringTexture, 5, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        descriptorSet->descriptors.push_back(vsg::DescriptorImage::create(model->_singleMieScatteringTexture, 6, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));

        descriptorSetLayout->bindings.push_back({2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        descriptorSetLayout->bindings.push_back({3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        descriptorSetLayout->bindings.push_back({4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        descriptorSetLayout->bindings.push_back({5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        descriptorSetLayout->bindings.push_back({6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    }

    void AtmosphereLighting::pack()
    {
        if(!directionalLights.empty())
        {

        }
    }

    AtmosphereLighting::~AtmosphereLighting()
    {
    }
}
