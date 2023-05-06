#ifndef ATMOSPHERELIGHTING_H
#define ATMOSPHERELIGHTING_H

#include "Atmosphere.h"
#include <vsg/state/ViewDependentState.h>
#include <vsg/utils/ShaderSet.h>

namespace atmosphere {

    class AtmosphereLighting : public vsg::Inherit<vsg::ViewDependentState, AtmosphereLighting>
    {
    public:
        AtmosphereLighting(vsg::ref_ptr<AtmosphereModel> model, uint32_t maxNumberLights = 64, uint32_t maxViewports = 1);

        vsg::ref_ptr<AtmosphereModel> atmosphereModel;

    protected:
        ~AtmosphereLighting();
    };
}

#endif // ATMOSPHERELIGHTING_H
