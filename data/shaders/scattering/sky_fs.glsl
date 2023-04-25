#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "rendering_constants.glsl"
#include "constants.glsl"
#include "functions.glsl"

layout(location = 0) in vec3 inRay;
layout(location = 0) out vec4 outColor;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(set = 0, binding = 0, std140) uniform Settings
{
	vec4 white_point_exp;
	vec4 sun_direction;
	vec2 sun_size;
} settings;

layout(set = 0, binding = 1) uniform sampler2D transmittance_texture;
layout(set = 0, binding = 2) uniform sampler2D irradiance_texture;
layout(set = 0, binding = 3) uniform sampler3D scattering_texture;
layout(set = 0, binding = 4) uniform sampler3D single_mie_scattering_texture;


layout(push_constant) uniform PushConstants {
    mat4 projection;
    mat4 modelView;
    mat4 invProjection;
    mat4 invModelView;
} pc;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------


Luminance3 GetSolarRadiance() {
      return solar_irradiance /
          (PI * sun_angular_radius * sun_angular_radius) *
          SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
Luminance3 GetSkyRadiance(
        Position camera, Direction view_ray, Length shadow_length,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadiance(transmittance_texture,
          scattering_texture, single_mie_scattering_texture,
          camera, view_ray, shadow_length, sun_direction, transmittance) *
          SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
    Luminance3 GetSkyRadianceToPoint(
        Position camera, Position point, Length shadow_length,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadianceToPoint(transmittance_texture,
          scattering_texture, single_mie_scattering_texture,
          camera, point, shadow_length, sun_direction, transmittance) *
          SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
    Illuminance3 GetSunAndSkyIrradiance(
       Position p, Direction normal, Direction sun_direction,
       out IrradianceSpectrum sky_irradiance) {
      IrradianceSpectrum sun_irradiance = GetSunAndSkyIrradiance(
          transmittance_texture, irradiance_texture, p, normal,
          sun_direction, sky_irradiance);
      sky_irradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
      return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }

// ------------------------------------------------------------------

// ------------------------------------------------------------------

void main()
{
	vec3 view_direction = normalize(inRay);

    vec3 cameraPos = pc.invModelView[3].xyz;

	// Compute the radiance of the sky.
	vec3 transmittance;
	vec3 radiance = GetSkyRadiance(cameraPos, view_direction, 4.0, settings.sun_direction.xyz, transmittance);

	// If the view ray intersects the Sun, add the Sun radiance.
	if (dot(view_direction, settings.sun_direction.xyz) > settings.sun_size.y)
		radiance = radiance + transmittance * GetSolarRadiance();

	outColor = vec4(pow(vec3(1.0) - exp(-radiance / settings.white_point_exp.rbg * settings.white_point_exp.a), vec3(1.0 / 2.2)), 1.0);
}
