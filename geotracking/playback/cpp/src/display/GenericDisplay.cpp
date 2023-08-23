#include "display/GenericDisplay.h"
#include "exporter/KMLExporter.h"

#ifdef COMPILE_WITH_ZEDHUB
#include <sl_hub/HubClient.hpp>
#endif

GenericDisplay::GenericDisplay()
{
}

GenericDisplay::~GenericDisplay()
{
#ifdef COMPILE_WITH_ZEDHUB
    sl_hub::STATUS_CODE exit_status = sl_hub::HubClient::disconnect();
    if (exit_status != sl_hub::STATUS_CODE::SUCCESS)
    {
        std::cout << "[ZedHub][ERROR] Terminate with error " << exit_status << std::endl;
        exit(EXIT_FAILURE);
    }
#else 
    closeAllKMLWriter();
#endif
}

void GenericDisplay::init(int argc, char **argv)
{
    opengl_viewer.init(argc, argv);

#ifdef COMPILE_WITH_ZEDHUB
    sl_hub::STATUS_CODE status_iot = sl_hub::HubClient::connect("geotracking");
    if (status_iot != sl_hub::STATUS_CODE::SUCCESS)
    {
        std::cout << "[ZedHub][ERROR] Initialization error " << status_iot << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

void GenericDisplay::updatePoseData(sl::Transform zed_rt, std::string str_t, std::string str_r, sl::POSITIONAL_TRACKING_STATE state)
{
    opengl_viewer.updateData(zed_rt, str_t, str_r, state);
}

bool GenericDisplay::isAvailable(){
    return opengl_viewer.isAvailable();
}

void GenericDisplay::updateGeoPoseData(sl::GeoPose geo_pose, sl::Timestamp current_timestamp)
{
#ifdef COMPILE_WITH_ZEDHUB
    sl_hub::json zedhub_message;
    zedhub_message["layer_type"] = "geolocation";
    zedhub_message["label"] = "Fused_position";
    zedhub_message["position"] = {
        {"latitude", geo_pose.latlng_coordinates.getLatitude(false)},
        {"longitude", geo_pose.latlng_coordinates.getLongitude(false)},
        {"altitude", geo_pose.latlng_coordinates.getAltitude()}};
    zedhub_message["epoch_timestamp"] = static_cast<uint64_t>(current_timestamp);
    sl_hub::HubClient::sendDataToPeers("geolocation", zedhub_message.dump());
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
#else
    static bool already_display_warning_message = false;
    if(!already_display_warning_message){
        already_display_warning_message = true;
        std::cerr << std::endl << "ZEDHub was not found ... the computed Geopose will be saved as KML file." << std::endl;
        std::cerr << "Results will be saved in  \"fused_position.kml\" file. You could use google myMaps (https://www.google.com/maps/about/mymaps/) to visualize it." << std::endl;
    }
    saveKMLData("fused_position.kml", geo_pose);
#endif
}
