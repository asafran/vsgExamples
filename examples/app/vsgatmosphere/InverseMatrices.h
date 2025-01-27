#ifndef INVERSEMATRICES_H
#define INVERSEMATRICES_H

#include <vsg/app/ViewMatrix.h>
#include <vsg/app/ProjectionMatrix.h>
#include <vsg/nodes/Transform.h>

namespace atmosphere {
    class InverseProjection : public vsg::Inherit<vsg::ProjectionMatrix, InverseProjection>
    {
    public:
        explicit InverseProjection(vsg::ref_ptr<vsg::ProjectionMatrix> pm)
            : projectionMatrix(pm)
        {
        }

        /// returns projectionMatrix->inverse()
        vsg::dmat4 transform() const override
        {
            return projectionMatrix->inverse();
        }

        void changeExtent(const VkExtent2D& prevExtent, const VkExtent2D& newExtent) override
        {
            projectionMatrix->changeExtent(prevExtent, newExtent);
        }
        vsg::ref_ptr<vsg::ProjectionMatrix> projectionMatrix;
    };

    class InverseView : public vsg::Inherit<vsg::ViewMatrix, InverseView>
    {
    public:
        explicit InverseView(double s, vsg::ref_ptr<vsg::ViewMatrix> vm)
            : viewMatrix(vm)
            , scale(s)
        {
        }

        /// returns projectionMatrix->inverse()
        vsg::dmat4 transform() const override
        {
            auto inverse = viewMatrix->inverse();
            inverse(3,0) *= scale;
            inverse(3,1) *= scale;
            inverse(3,2) *= scale;
            return inverse;
        }

        vsg::ref_ptr<vsg::ViewMatrix> viewMatrix;
        double scale;
    };

    class InverseTransform : public vsg::Inherit<vsg::Transform, InverseTransform>
    {
    public:
        InverseTransform() {}
        explicit InverseTransform(const vsg::dmat4& in_matrix) : matrix(in_matrix) {}

        vsg::dmat4 transform(const vsg::dmat4& mv) const override { return vsg::inverse(mv * matrix); }

        vsg::dmat4 matrix;

    protected:
    };
}
#endif // INVERSEMATRICES_H
