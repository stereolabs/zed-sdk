#include "display/GenericDisplay.h"
#include "exporter/KMLExporter.h"


GenericDisplay::GenericDisplay()
{
}

GenericDisplay::~GenericDisplay()
{
    closeAllKMLWriter();
}

void GenericDisplay::init(int argc, char **argv)
{
    opengl_viewer.init(argc, argv);
}

void GenericDisplay::updatePoseData(sl::Transform zed_rt, sl::FusedPositionalTrackingStatus state)
{
    opengl_viewer.updateData(zed_rt, state);
}

bool GenericDisplay::isAvailable(){
    return opengl_viewer.isAvailable();
}

void GenericDisplay::updateRawGeoPoseData(sl::GNSSData geo_data)
{
    double latitude, longitude, altitude;
    geo_data.getCoordinates(latitude, longitude, altitude, false);

    // Make the pose available for the Live Server
    ofstream data;
    data.open ("../../../map server/raw_data.txt");
    data << std::fixed << std::setprecision(17);
    data << latitude;
    data << ",";
    data << longitude;
    data << ",";
    data << geo_data.ts.getMilliseconds();
    data << "\n";
    data.close();
}

void GenericDisplay::updateGeoPoseData(sl::GeoPose geo_pose, sl::Timestamp current_timestamp)
{
    // Make the pose available for the Live Server
    ofstream data;
    data.open ("../../../map server/data.txt");
    data << std::fixed << std::setprecision(17);
    data << geo_pose.latlng_coordinates.getLatitude(false);
    data << ",";
    data << geo_pose.latlng_coordinates.getLongitude(false);
    data << ",";
    data << current_timestamp.getMilliseconds();
    data << "\n";
    data.close();

    // Save the pose in a .kml file
    saveKMLData("fused_position.kml", geo_pose);
}
