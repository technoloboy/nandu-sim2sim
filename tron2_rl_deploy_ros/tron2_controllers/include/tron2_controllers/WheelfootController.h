#pragma once

#include "tron2_controllers/ControllerBase.h"

namespace tron2_controller
{

class WheelfootController : public ControllerBase
{
protected:
    void computeObservation() override;
    void applyActions() override;
};

} // namespace tron2_controller
