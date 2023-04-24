#include "display/GenericDisplay.h"

#ifdef COMPILE_WITH_ZEDHUB
#include <sl_iot/HubClient.hpp>
#endif

GenericDisplay::GenericDisplay()
{
}

GenericDisplay::~GenericDisplay()
{
#ifdef COMPILE_WITH_ZEDHUB
    sl_iot::STATUS_CODE exit_sl_iot_status = sl_iot::HubClient::disconnect();
    if (exit_sl_iot_status != sl_iot::STATUS_CODE::SUCCESS)
    {
        std::cout << "[ZedHub][ERROR] Terminate with error " << exit_sl_iot_status << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

void GenericDisplay::init(int argc, char **argv)
{
    opengl_viewer.init(argc, argv);

#ifdef COMPILE_WITH_ZEDHUB
    sl_iot::STATUS_CODE status_iot = sl_iot::HubClient::connect("geotracking");
    if (status_iot != sl_iot::STATUS_CODE::SUCCESS)
    {
        std::cout << "[ZedHub][ERROR] Initialization error " << status_iot << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

void GenericDisplay::updatePoseData(sl::Transform zed_rt, sl::POSITIONAL_TRACKING_STATE state)
{
    opengl_viewer.updateData(zed_rt, state);
}

bool GenericDisplay::isAvailable(){
    return opengl_viewer.isAvailable();
}

void GenericDisplay::updateGeoPoseData(sl::GeoPose geo_pose, sl::Timestamp current_timestamp)
{
#ifdef COMPILE_WITH_ZEDHUB
    sl_iot::json zedhub_message;
    zedhub_message["layer_type"] = "geolocation";
    zedhub_message["label"] = "Fused_position";
    zedhub_message["position"] = {
        {"latitude", geo_pose.latlng_coordinates.getLatitude(false)},
        {"longitude", geo_pose.latlng_coordinates.getLongitude(false)},
        {"altitude", geo_pose.latlng_coordinates.getAltitude()}};
    zedhub_message["epoch_timestamp"] = static_cast<uint64_t>(current_timestamp);
    sl_iot::HubClient::sendDataToPeers("geolocation", zedhub_message.dump());
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
#else
    static bool already_display_warning_message = false;
    if(!already_display_warning_message){
        already_display_warning_message = true;
        std::cerr << "ZEDHub was not found ... the computed Geopose will be saved as KML file" << std::endl;
    }
#endif
}
