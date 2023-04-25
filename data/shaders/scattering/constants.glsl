/**
 * Copyright (c) 2017 Eric Bruneton
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This GLSL file defines the physical types and constants which are used in the
 * main functions of our atmosphere model.
 *
 * Physical quantities
 *
 * The physical quantities we need for our atmosphere model are radiometric and photometric quantities.
 * 
 * https://en.wikipedia.org/wiki/Radiometry
 * https://en.wikipedia.org/wiki/Photometry_(optics)
 *
 * We start with six base quantities: length, wavelength, angle, solid angle,
 * power and luminous power (wavelength is also a length, but we distinguish the
 * two for increased clarity).
 *
 */

layout (constant_id = 10) const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
layout (constant_id = 11) const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;

layout (constant_id = 12) const int SCATTERING_TEXTURE_R_SIZE = 32;
layout (constant_id = 13) const int SCATTERING_TEXTURE_MU_SIZE = 128;
layout (constant_id = 14) const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
layout (constant_id = 15) const int SCATTERING_TEXTURE_NU_SIZE = 8;

const int SCATTERING_TEXTURE_WIDTH = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
const int SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
const int SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;

layout (constant_id = 16) const int IRRADIANCE_TEXTURE_WIDTH = 64;
layout (constant_id = 17) const int IRRADIANCE_TEXTURE_HEIGHT = 16;

/*
layout (constant_id = 24) const float aX = 0.0f;
layout (constant_id = 25) const float aY = 0.0f;
layout (constant_id = 26) const float aZ = 0.0f;
//const vec3 absorption_extinction = vec3(aX, aY, aZ);

layout (constant_id = 30) const float mieexX = 0.0f;
layout (constant_id = 31) const float mieexY = 0.0f;
layout (constant_id = 32) const float mieexZ = 0.0f;
//const vec3 mie_extinction = vec3(mieexX, mieexY, mieexZ);

layout (constant_id = 33) const float gX = 0.0f;
layout (constant_id = 34) const float gY = 0.0f;
layout (constant_id = 35) const float gZ = 0.0f;
//const vec3 ground_albedo = vec3(gX, gY, gZ);
*/

layout (constant_id = 36) const float sun_angular_radius = 0.0f;
layout (constant_id = 37) const float bottom_radius = 0.0f;
layout (constant_id = 38) const float top_radius = 0.0f;
layout (constant_id = 39) const float mie_phase_function_g = 0.0f;
layout (constant_id = 40) const float mu_s_min = 0.0f;
/*
layout (constant_id = 47) const float rayleigh_width = 0.0f;
layout (constant_id = 48) const float rayleigh_exp_term = 0.0f;
layout (constant_id = 49) const float rayleigh_exp_scale = 0.0f;
layout (constant_id = 50) const float rayleigh_linear_term = 0.0f;
layout (constant_id = 51) const float rayleigh_constant_term = 0.0f;

layout (constant_id = 52) const float mie_width = 0.0f;
layout (constant_id = 53) const float mie_exp_term = 0.0f;
layout (constant_id = 54) const float mie_exp_scale = 0.0f;
layout (constant_id = 55) const float mie_linear_term = 0.0f;
layout (constant_id = 56) const float mie_constant_term = 0.0f;

layout (constant_id = 57) const float absorption0_width = 0.0f;
layout (constant_id = 58) const float absorption0_exp_term = 0.0f;
layout (constant_id = 59) const float absorption0_exp_scale = 0.0f;
layout (constant_id = 60) const float absorption0_linear_term = 0.0f;
layout (constant_id = 61) const float absorption0_constant_term = 0.0f;

layout (constant_id = 62) const float absorption1_width = 0.0f;
layout (constant_id = 63) const float absorption1_exp_term = 0.0f;
layout (constant_id = 64) const float absorption1_exp_scale = 0.0f;
layout (constant_id = 65) const float absorption1_linear_term = 0.0f;
layout (constant_id = 66) const float absorption1_constant_term = 0.0f;
*/
#define Length float
#define Wavelength float
#define Angle float
#define SolidAngle float
#define Power float
#define LuminousPower float

#define assert(x)
#define IN(x) const in x
#define OUT(x) out x

#define TEMPLATE(x)

/*
 * From this we "derive" the irradiance, radiance, spectral irradiance,
 * spectral radiance, luminance, etc, as well pure numbers, area, volume.
 */

#define Number float
#define InverseLength float
#define Area float
#define Volume float
#define NumberDensity float
#define Irradiance float
#define Radiance float
#define SpectralPower float
#define SpectralIrradiance float
#define SpectralRadiance float
#define SpectralRadianceDensity float
#define ScatteringCoefficient float
#define InverseSolidAngle float
#define LuminousIntensity float
#define Luminance float
#define Illuminance float

/*
 * We  also need vectors of physical quantities, mostly to represent functions
 * depending on the wavelength. In this case the vector elements correspond to
 * values of a function at some predefined wavelengths.
 */

// A generic function from Wavelength to some other type.
#define AbstractSpectrum vec3
// A function from Wavelength to Number.
#define DimensionlessSpectrum vec3
// A function from Wavelength to SpectralPower.
#define PowerSpectrum vec3
// A function from Wavelength to SpectralIrradiance.
#define IrradianceSpectrum vec3
// A function from Wavelength to SpectralRadiance.
#define RadianceSpectrum vec3
// A function from Wavelength to SpectralRadianceDensity.
#define RadianceDensitySpectrum vec3
// A function from Wavelength to ScaterringCoefficient.
#define ScatteringSpectrum vec3

// A position in 3D (3 length values).
#define Position vec3
// A unit direction vector in 3D (3 unitless values).
#define Direction vec3
// A vector of 3 luminance values.
#define Luminance3 vec3
// A vector of 3 illuminance values.
#define Illuminance3 vec3

/*
 * Finally, we also need precomputed textures containing physical quantities in each texel.
 */

#define TransmittanceTexture sampler2D
#define AbstractScatteringTexture sampler3D
#define ReducedScatteringTexture sampler3D
#define ScatteringTexture sampler3D
#define ScatteringDensityTexture sampler3D
#define IrradianceTexture sampler2D

/*
 * Physical units
 *
 * We can then define the units for our six base physical quantities:
 * meter (m), nanometer (nm), radian (rad), steradian (sr), watt (watt) and lumen (lm):
 */

const Length m = 1.0;
const Wavelength nm = 1.0;
const Angle rad = 1.0;
const SolidAngle sr = 1.0;
const Power watt = 1.0;
const LuminousPower lm = 1.0;

/*
 * From which we can derive the units for some derived physical quantities,
 * as well as some derived units (kilometer km, kilocandela kcd, degree deg):
 */

const float PI = 3.14159265358979323846;

const Length km = 1000.0 * m;
const Area m2 = m * m;
const Volume m3 = m * m * m;
const Angle pi = PI * rad;
const Angle deg = pi / 180.0;
const Irradiance watt_per_square_meter = watt / m2;
const Radiance watt_per_square_meter_per_sr = watt / (m2 * sr);
const SpectralIrradiance watt_per_square_meter_per_nm = watt / (m2 * nm);
const SpectralRadiance watt_per_square_meter_per_sr_per_nm = watt / (m2 * sr * nm);
const SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm = watt / (m3 * sr * nm);
const LuminousIntensity cd = lm / sr;
const LuminousIntensity kcd = 1000.0 * cd;
const Luminance cd_per_square_meter = cd / m2;
const Luminance kcd_per_square_meter = kcd / m2;

/*
 * Atmosphere parameters
 *
 * Using the above types, we can now define the parameters of our atmosphere
 * model. We start with the definition of density profiles, which are needed for
 * parameters that depend on the altitude:
 */


/*
layout(std430, set = 0, binding = 1) buffer DensityProfileLayerUbo
{
    DensityProfileLayer[4] layers;
};
*/

// An atmosphere density profile made of several layers on top of each other
// (from bottom to top). The width of the last layer is ignored, i.e. it always
// extend to the top atmosphere boundary. The profile values vary between 0
// (null density) to 1 (maximum density).
/*struct DensityProfile
{
   DensityProfileLayer layers[2];
};

DensityProfile RayleighDensity()
{
  DensityProfileLayer layer;

  layer.width = rayleigh_width;
  layer.exp_term = rayleigh_exp_term;
  layer.exp_scale = rayleigh_exp_scale;
  layer.linear_term = rayleigh_linear_term;
  layer.constant_term = rayleigh_constant_term;

  DensityProfile profile;
  profile.layers[0] = layer;
  profile.layers[1] = layer;

  return profile;
}

DensityProfile MieDensity()
{
  DensityProfileLayer layer;

  layer.width = mie_width;
  layer.exp_term = mie_exp_term;
  layer.exp_scale = mie_exp_scale;
  layer.linear_term = mie_linear_term;
  layer.constant_term = mie_constant_term;

  DensityProfile profile;
  profile.layers[0] = layer;
  profile.layers[1] = layer;

  return profile;
}

DensityProfile AbsorptionDensity()
{
  DensityProfileLayer layer0;

  layer0.width = absorption0_width;
  layer0.exp_term = absorption0_exp_term;
  layer0.exp_scale = absorption0_exp_scale;
  layer0.linear_term = absorption0_linear_term;
  layer0.constant_term = absorption0_constant_term;

  DensityProfileLayer layer1;

  layer1.width = absorption1_width;
  layer1.exp_term = absorption1_exp_term;
  layer1.exp_scale = absorption1_exp_scale;
  layer1.linear_term = absorption1_linear_term;
  layer1.constant_term = absorption1_constant_term;

  DensityProfile profile;
  profile.layers[0] = layer0;
  profile.layers[1] = layer1;

  return profile;
}
*/
