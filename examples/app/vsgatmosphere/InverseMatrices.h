#ifndef INVERSEMATRICES_H
#define INVERSEMATRICES_H

#include <vsg/app/ViewMatrix.h>
#include <vsg/app/ProjectionMatrix.h>

namespace atmosphere {

    class InversePerspective : public vsg::Inherit<vsg::EllipsoidPerspective, InversePerspective>
    {
    public:

        InversePerspective(vsg::ref_ptr<vsg::LookAt> la, vsg::ref_ptr<vsg::EllipsoidModel> em, vsg::ref_ptr<vsg::mat4Array> inverseMatrices)
            : vsg::Inherit<vsg::EllipsoidPerspective, InversePerspective>(la, em)
            , _data(inverseMatrices)
        {
            _data->set(0, vsg::mat4(inverse()));
        }

        InversePerspective(vsg::ref_ptr<vsg::LookAt> la, vsg::ref_ptr<vsg::EllipsoidModel> em, vsg::ref_ptr<vsg::mat4Array> inverseMatrices,
                           double fov, double ar, double nfr, double hmh)
            : vsg::Inherit<vsg::EllipsoidPerspective, InversePerspective>(la, em, fov, ar, nfr, hmh)
            , _data(inverseMatrices)
        {
            _data->set(0, vsg::mat4(inverse()));
        }

        void changeExtent(const VkExtent2D& prevExtent, const VkExtent2D& newExtent) override
        {
            vsg::EllipsoidPerspective::changeExtent(prevExtent, newExtent);
            _data->set(0, vsg::mat4(inverse()));
        }
    protected:
        vsg::ref_ptr<vsg::mat4Array> _data;
    };

    class InverseView : public vsg::Inherit<vsg::LookAt, InverseView>
    {
    public:

        InverseView(vsg::ref_ptr<vsg::mat4Array> inverseMatrices, vsg::ref_ptr<vsg::vec4Value> camera)
            : vsg::Inherit<vsg::LookAt, InverseView>()
            , _matrices(inverseMatrices)
            , _camera(camera)
        {
            update();
        }

        void update()
        {
            auto inverseMatrix = inverse();
            inverseMatrix(3, 0) /= 1000.0;
            inverseMatrix(3, 1) /= 1000.0;
            inverseMatrix(3, 2) /= 1000.0;
            _matrices->set(1, vsg::mat4(inverseMatrix));
            _camera->set(vsg::vec4(eye.x / 1000.0, eye.y / 1000.0, eye.z / 1000.0, 0.0f));
        }
    protected:
        vsg::ref_ptr<vsg::mat4Array> _matrices;
        vsg::ref_ptr<vsg::vec4Value> _camera;
    };
}
#endif // INVERSEMATRICES_H
