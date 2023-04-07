#ifndef KML_EXPORTER_H
#define KML_EXPORTER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sl/Fusion.hpp>

/**
 * @brief Save GeoPose in KML file that can be displayed into google map (maps.google.com)
 * 
 * @param file_path path expected for the resulted KML file
 * @param geopose current data to save
 */
void saveKMLData(std::string file_path, sl::GeoPose geopose);
/**
 * @brief Save GNSSData in KML file that can be displayed into google map (maps.google.com)
 * 
 * @param file_path path expected for the resulted KML file
 * @param gnss_data current data to save
 */
void saveKMLData(std::string file_path, sl::GNSSData gnss_data);
/**
 * @brief Close all KML file writer and place KML files footer
 * 
 */
void closeAllKMLWriter();

#endif