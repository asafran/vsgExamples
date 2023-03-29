#ifndef ATMOSPHERETOOLS_H
#define ATMOSPHERETOOLS_H

#include "atmoshpere-constatnts.h"
#include <vector>
#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>
namespace atmosphere {

double interpolate(const std::vector<double>& wavelengths, const std::vector<double>& wavelength_function, double wavelength)
{
    if (wavelength < wavelengths[0])
        return wavelength_function[0];

    for (size_t i = 0; i < wavelengths.size() - 1; ++i)
    {
        if (wavelength < wavelengths[i + 1])
        {
            double u = (wavelength - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i]);
            return wavelength_function[i] * (1.0 - u) + wavelength_function[i + 1] * u;
        }
    }

    return wavelength_function[wavelength_function.size() - 1];
}

constexpr double cie_color_matching_function_table_value(double wavelength, int column)
{
    if (wavelength <= kLambdaMin || wavelength >= kLambdaMax)
        return 0.0;

    double u = (wavelength - kLambdaMin) / 5.0;
    int row = (int)floor(u);

    u -= row;
    return kCIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row + column] * (1.0 - u) + kCIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1) + column] * u;
}

vsg::vec4 compute_spectral_radiance_to_luminance_factors(const std::vector<double>& wavelengths, const std::vector<double>& solar_irradiance, double lambda_power)
{
    double k_r = 0.0;
    double k_g = 0.0;
    double k_b = 0.0;
    double solar_r = interpolate(wavelengths, solar_irradiance, kLambdaR);
    double solar_g = interpolate(wavelengths, solar_irradiance, kLambdaG);
    double solar_b = interpolate(wavelengths, solar_irradiance, kLambdaB);
    int dlambda = 1;

    for (int lambda = kLambdaMin; lambda < kLambdaMax; lambda += dlambda)
    {
        double x_bar = cie_color_matching_function_table_value(lambda, 1);
        double y_bar = cie_color_matching_function_table_value(lambda, 2);
        double z_bar = cie_color_matching_function_table_value(lambda, 3);

        const double* xyz2srgb = &kXYZ_TO_SRGB[0];
        double r_bar = xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
        double g_bar = xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
        double b_bar = xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
        double irradiance = interpolate(wavelengths, solar_irradiance, lambda);

        k_r += r_bar * irradiance / solar_r * pow(lambda / kLambdaR, lambda_power);
        k_g += g_bar * irradiance / solar_g * pow(lambda / kLambdaG, lambda_power);
        k_b += b_bar * irradiance / solar_b * pow(lambda / kLambdaB, lambda_power);
    }

    k_r *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;
    k_g *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;
    k_b *= static_cast<double>(MAX_LUMINOUS_EFFICACY) * dlambda;

    return {static_cast<float>(k_r), static_cast<float>(k_g), static_cast<float>(k_b), 0.0f};
}

vsg::vec4 to_vector(const std::vector<double>& wavelengths, const std::vector<double>& v, double scale)
{
    double r = interpolate(wavelengths, v, kLambdaR) * scale;
    double g = interpolate(wavelengths, v, kLambdaG) * scale;
    double b = interpolate(wavelengths, v, kLambdaB) * scale;

    return {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b), 0.0f};
}

constexpr vsg::mat4 to_matrix(double arr[])
{
    return vsg::mat4(
        (float)arr[0], (float)arr[3], (float)arr[6], 0.0f,
        (float)arr[1], (float)arr[4], (float)arr[7], 0.0f,
        (float)arr[2], (float)arr[5], (float)arr[8], 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

constexpr double coeff(double lambda, int component)
{
    // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid
    // artefacts due to too large values when using half precision on GPU.
    // We add this term back in kAtmosphereShader, via
    // SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
    // Model constructor).
    double x = cie_color_matching_function_table_value(lambda, 1);
    double y = cie_color_matching_function_table_value(lambda, 2);
    double z = cie_color_matching_function_table_value(lambda, 3);
    double sRGB = kXYZ_TO_SRGB[component * 3 + 0] * x + kXYZ_TO_SRGB[component * 3 + 1] * y + kXYZ_TO_SRGB[component * 3 + 2] * z;

    return sRGB;
}
}

#endif // ATMOSPHERETOOLS_H
