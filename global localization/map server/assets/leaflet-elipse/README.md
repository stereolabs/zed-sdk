Leaflet.ellipse
===============

A basic ellipse type for [Leaflet](http://leafletjs.com), a JS 
library for interactive maps.  Allows specification of semi-major and
semi-minor axis as well as a tilt degrees from west.

*Supports Leaflet 1.0.0 or newer.*

## Supported browsers

* Internet Explorer 10+
* Google Chrome
* Safari
* Firefox 17+

## Demo

A demo is available on the Github Pages webpage for Leaflet.ellipse [here](http://jdfergason.github.io/Leaflet.Ellipse/).

## Usage

Include the ellipse javasript file:

    <script src="l.ellipse.js"></script>

After instantiating the map create a new ellipse.

    var ellipse = L.ellipse([51.5, -0.09], [500, 100], 90).addTo(map);

## API

*Factory method*

    L.ellipse( <LatLng> latlng, <Radii> radii, <Number> tilt, 
               <Path options> options? )

 * latlng  - The position of the center of the ellipse.
 * radii   - The semi-major and semi-minor axis in meters
 * tilt    - The rotation of the ellipse in degrees from west
 * options - Options dictionary to pass to L.Path

## License

This code is provided under the Apache 2.0 license.
