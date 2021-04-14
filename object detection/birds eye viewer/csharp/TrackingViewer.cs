//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using sl;

using OpenCvSharp;

public enum TrackPointState
{
    OK,
    PREDICTED,
    OFF
}

public struct TrackPoint
{
    public float x, y, z;
    public ulong timestamp;
    public TrackPointState tracking_state;

    public TrackPoint(Vector3 pos, TrackPointState state, ulong ts)
    {
        x = pos.X;
        y = pos.Y;
        z = pos.Z;
        tracking_state = state;
        timestamp = ts;
    }

    public TrackPoint(Vector3 pos, sl.OBJECT_TRACKING_STATE state, ulong ts)
    {
        x = pos.X;
        y = pos.Y;
        z = pos.Z;

        if (state == OBJECT_TRACKING_STATE.OK) tracking_state = TrackPointState.OK;
        else tracking_state = TrackPointState.OFF;

        timestamp = ts;
    }

    public Vector3 toVector3()
    {
        return new Vector3(x, y, z);
    }
};

public class Tracklet
{
    public int id;
    public List<TrackPoint> positions = new List<TrackPoint>(); // Will store detected positions and the predicted ones
    public List<TrackPoint> positions_to_draw = new List<TrackPoint>(); // Will store the visualization output => when smoothing track, point won't be the same as the real points

    public sl.OBJECT_TRACKING_STATE tracking_state;
    public sl.OBJECT_CLASS object_type;
    public ulong last_detected_timestamp;
    public int recovery_cpt;

    // Track state
    public bool is_alive;

    private static int recovery_length = 10;

    public Tracklet(sl.ObjectData obj, sl.OBJECT_CLASS type, ulong ts = 0)
    {
        id = obj.id;
        positions.Add(new TrackPoint(obj.position, obj.objectTrackingState, ts));
        positions_to_draw.Add(new TrackPoint(obj.position, obj.objectTrackingState, ts));
        tracking_state = obj.objectTrackingState;
        last_detected_timestamp = ts;
        recovery_cpt = recovery_length;
        is_alive = true;
        object_type = type;
    }

    public void addDetectedPoint(sl.ObjectData obj, ulong ts, int smoothing_window_size = 0)
    {
        if (positions.Count > 0)
        {
            if (positions[positions.Count - 1].tracking_state == TrackPointState.PREDICTED) recovery_cpt = 0;
            else ++recovery_cpt;
        }

        positions.Add(new TrackPoint(obj.position, TrackPointState.OK, ts));
        tracking_state = obj.objectTrackingState;
        last_detected_timestamp = ts;


        positions_to_draw.Add(new TrackPoint(obj.position, TrackPointState.OK, ts));
    }
}

public class TrackingViewer
{
    float x_min, x_max;  // show objects between [x_min; x_max] (in millimeters) 
    float x_step, z_step;     // Conversion from world position to pixel coordinates
    float z_min; // show objects between [z_min; 0] (z_min < 0) (in millimeters)

    // window size
    int window_width, window_height;
    // Keep tracks of alive tracks
    List<Tracklet> tracklets = new List<Tracklet>();

    //History management
    ulong history_duration; // in ns
    int min_length_to_draw;

    // Visualization configuration
    OpenCvSharp.Mat background;
    bool has_background_ready;
    OpenCvSharp.Scalar background_color, fov_color;
    int camera_offset;

    // Camera Settings
    sl.CalibrationParameters camera_calibration;
    float fov;

    // SMOOTH
    bool do_smooth;
    int smoothing_window_size;


    static sl.float2 getImagePosition(Vector2[] bounding_box_image, sl.float2 img_scale) {
        sl.float2 position;
        position.x = (bounding_box_image[0].X + (bounding_box_image[2].X - bounding_box_image[0].X)*0.5f) * img_scale.x;
        position.y = (bounding_box_image[0].Y + (bounding_box_image[2].Y - bounding_box_image[0].Y)*0.5f) * img_scale.y;
        return position;
    }

    public static void render_2D(ref OpenCvSharp.Mat left_display, sl.float2 img_scale, ref sl.Objects objects, bool render_mask, bool isTrackingON)
    {
        OpenCvSharp.Mat overlay = left_display.Clone();
        OpenCvSharp.Rect roi_render = new OpenCvSharp.Rect(0, 0, left_display.Size().Width, left_display.Size().Height);

        OpenCvSharp.Mat mask = new OpenCvSharp.Mat(left_display.Rows, left_display.Cols, OpenCvSharp.MatType.CV_8UC1);

        int line_thickness = 2;

        for (int i = 0; i < objects.numObject; i++)
        {
            sl.ObjectData obj = objects.objectData[i];
            if (Utils.renderObject(obj, isTrackingON))
            {
                OpenCvSharp.Scalar base_color = Utils.generateColorID_u(obj.id);

                // Display image scale bouding box 2d
                if (obj.boundingBox2D.Length < 4) continue;

                Point top_left_corner = Utils.cvt(obj.boundingBox2D[0], img_scale);
                Point top_right_corner = Utils.cvt(obj.boundingBox2D[1], img_scale);
                Point bottom_right_corner = Utils.cvt(obj.boundingBox2D[2], img_scale);
                Point bottom_left_corner = Utils.cvt(obj.boundingBox2D[3], img_scale);

                // Create of the 2 horizontal lines
                Cv2.Line(left_display, top_left_corner, top_right_corner, base_color, line_thickness);
                Cv2.Line(left_display, bottom_left_corner, bottom_right_corner, base_color, line_thickness);
                // Creation of two vertical lines
                Utils.drawVerticalLine(ref left_display, bottom_left_corner, top_left_corner, base_color, line_thickness);
                Utils.drawVerticalLine(ref left_display, bottom_right_corner, top_right_corner, base_color, line_thickness);

                // Scaled ROI
                OpenCvSharp.Rect roi = new OpenCvSharp.Rect(top_left_corner.X, top_left_corner.Y, (int)top_right_corner.DistanceTo(top_left_corner), (int)bottom_right_corner.DistanceTo(top_right_corner));

                overlay.SubMat(roi).SetTo(base_color);

                sl.float2 position_image = getImagePosition(obj.boundingBox2D, img_scale);
                Cv2.PutText(left_display, obj.label.ToString(), new Point(position_image.x - 20, position_image.y - 12), HersheyFonts.HersheyComplexSmall, 0.5f, new Scalar(255, 255, 255, 255), 1);

                if (!float.IsInfinity(obj.position.Z))
                {
                    string text = Math.Abs(obj.position.Z).ToString("0.##M");
                    Cv2.PutText(left_display, text, new Point(position_image.x - 20, position_image.y), HersheyFonts.HersheyComplexSmall, 0.5, new Scalar(255, 255, 255, 255), 1);
                }
            }
        }

        // Here, overlay is as the left image, but with opaque masks on each detected objects
        Cv2.AddWeighted(left_display, 0.7, overlay, 0.3, 0.0, left_display);
    }

    public TrackingViewer()
    {
    }

    public TrackingViewer(sl.Resolution res, int fps_, float D_max, int duration)
    {
        // ----------- Default configuration -----------------

        // window size
        window_width = (int)res.width;
        window_height = (int)res.height;

        // Visualization configuration
        camera_offset = 50;

        // history management
        min_length_to_draw = 3;

        // camera settings
        fov = -1.0f;

        // Visualization settings
        background_color = new Scalar(248, 248, 248, 255);
        has_background_ready = false;
        background = new OpenCvSharp.Mat(window_height, window_width, MatType.CV_8UC4, background_color);

        Scalar ref_ = new Scalar(255, 117, 44, 255);

        for (int p = 0; p < 3; p++)
            fov_color[p] = (ref_[p] + 2 * background_color[p]) / 3;

        // SMOOTH
        do_smooth = false;

        // Show last 3.0 seconds
        history_duration = (ulong)(duration) * 1000 * 1000 * 1000; //convert sc to ns

        // Smoothing window: 80ms
        smoothing_window_size = (int)(Math.Ceiling(0.08f * fps_) + .5f);

        // invert Z due to Y axis of ocv windows
        z_min = -D_max;
        x_min = z_min / 2.0f;
        x_max = -x_min;

        x_step = (x_max - x_min) / window_width;
        z_step = Math.Abs(z_min) / (window_height - camera_offset);
    }
   
    public void generate_view(ref sl.Objects objects, sl.Pose current_camera_pose, ref OpenCvSharp.Mat tracking_view, bool tracking_enabled)
    {
        // To get position in WORLD reference
        for (int i = 0; i < objects.numObject; i++){
            sl.ObjectData obj = objects.objectData[i];

            Vector3 pos = obj.position;
            Vector3 new_pos =  Vector3.Transform(pos, current_camera_pose.rotation) + current_camera_pose.translation;
            obj.position = new_pos;
        }

        // Initialize visualization
        if (!has_background_ready)
            generateBackground();

        background.CopyTo(tracking_view);
        // Scale
        drawScale(ref tracking_view);

        if (tracking_enabled)
        {
            // First add new points, and remove the ones that are too old
            ulong current_timestamp = objects.timestamp;
            addToTracklets(ref objects);
            detectUnchangedTrack(current_timestamp);
            pruneOldPoints(current_timestamp);

            // Draw all tracklets
            drawTracklets(ref tracking_view, current_camera_pose);
        }
        else
        {
            drawPosition(ref objects, ref tracking_view, current_camera_pose);
        }
    }

    public void setCameraCalibration(sl.CalibrationParameters calib)
    {
        camera_calibration = calib;
        has_background_ready = false;
    }

    //Zoom functions
    public void zoomIn()
    {
        zoom(0.9f);
    }

    public void zoomOut()
    {
        zoom(1.0f / 0.9f);
    }

    // vizualisation methods
    void drawTracklets(ref OpenCvSharp.Mat tracking_view, sl.Pose current_camera_pose)
    {
        foreach (Tracklet track in tracklets) {
            if (track.tracking_state != sl.OBJECT_TRACKING_STATE.OK)
            {
                Console.WriteLine("not ok");
                continue;
            }
            if (track.positions_to_draw.Count < min_length_to_draw)
            {
                //Console.WriteLine("too small " + track.positions_to_draw.Count);
                continue;
            }

            Scalar clr = Utils.generateColorID_u((int)track.id);

            int track_size = track.positions_to_draw.Count;
            TrackPoint start_point = track.positions_to_draw[0];
            Point cv_start_point = toCVPoint(start_point, current_camera_pose);
            TrackPoint end_point = track.positions_to_draw[0];
            for (int point_index = 1; point_index < track_size; ++point_index)
            {
                end_point = track.positions_to_draw[point_index];
                Point cv_end_point = toCVPoint(track.positions_to_draw[point_index], current_camera_pose);

                // Check point status
                if (start_point.tracking_state == TrackPointState.OFF || end_point.tracking_state == TrackPointState.OFF)
                {
                    continue;
                }

                Cv2.Line(tracking_view, cv_start_point, cv_end_point, clr, 4);
                start_point = end_point;
                cv_start_point = cv_end_point;
            }

            // Current position, visualized as a point, only for alived track
            // Point = person || Square = Vehicle 
            if (track.is_alive)
            {
                switch (track.object_type)
                {
                    case sl.OBJECT_CLASS.PERSON:
                        Cv2.Circle(tracking_view, toCVPoint(track.positions_to_draw[track.positions_to_draw.Count - 1], current_camera_pose), 5, clr, 5);
                        break;
                    case sl.OBJECT_CLASS.VEHICLE:
                        {
                            Point rect_center = toCVPoint(track.positions_to_draw[track.positions_to_draw.Count - 1], current_camera_pose);
                            int square_size = 10;
                            Point top_left_corner = rect_center - new Point(square_size, square_size * 2);
                            Point right_bottom_corner = rect_center + new Point(square_size, square_size * 2);
                            Cv2.Rectangle(tracking_view, top_left_corner, right_bottom_corner, clr, Cv2.FILLED);

                            break;
                        }
                    case sl.OBJECT_CLASS.LAST:
                        break;
                    default:
                        break;
                }
            }
        }
    }

    void drawPosition(ref sl.Objects objects, ref OpenCvSharp.Mat tracking_view, sl.Pose current_camera_pose){
        for (int i = 0; i < objects.numObject; i++)
        {
            sl.ObjectData obj = objects.objectData[i];
            Scalar generated_color = Utils.generateColorClass_u((int)obj.label);

            // Point = person || Rect = Vehicle 
            switch (obj.label)
            {
                case sl.OBJECT_CLASS.PERSON:
                    Cv2.Circle(tracking_view, toCVPoint(obj.position, current_camera_pose), 5, generated_color, 5);
                    break;
                case sl.OBJECT_CLASS.VEHICLE:
                    {
                        if (obj.boundingBox.Length > 0)
                        {
                            Point rect_center = toCVPoint(obj.position, current_camera_pose);
                            int square_size = 10;
                            Point top_left_corner = rect_center - new Point(square_size, square_size * 2);
                            Point right_bottom_corner = rect_center + new Point(square_size, square_size * 2);

                            Cv2.Rectangle(tracking_view, top_left_corner, right_bottom_corner, generated_color, Cv2.FILLED);
                        }
                        break;
                    }
                case sl.OBJECT_CLASS.LAST:
                    break;
                default:
                    break;
            }
        }
    }

    void drawScale(ref OpenCvSharp.Mat tracking_view)
    {
        int one_meter_horizontal = (int)(1.0f / x_step + .5f);
        Point st_pt = new Point(25, window_height - 50);
        Point end_pt = new Point(25 + one_meter_horizontal, window_height - 50);
        int thickness = 1;

        // Scale line
        Cv2.Line(tracking_view, st_pt, end_pt, new Scalar(0, 0, 0, 255), thickness);

        // Add ticks
        Cv2.Line(tracking_view, st_pt + new Point(0, -3), st_pt + new Point(0, 3), new Scalar(0, 0, 0, 255), thickness);
        Cv2.Line(tracking_view, end_pt + new Point(0, -3), end_pt + new Point(0, 3), new Scalar(0, 0, 0, 255), thickness);

        // Scale text
        Cv2.PutText(tracking_view, "1m", end_pt + new Point(5, 5), HersheyFonts.HersheyPlain, 1.0, new Scalar(0, 0, 0, 255), 1);
    }

    void addToTracklets(ref sl.Objects objects)
    {
        ulong current_timestamp = objects.timestamp;
        for (int i = 0; i < objects.numObject; i++)
        {
            sl.ObjectData obj = objects.objectData[i];
            int id = obj.id;

            if ((obj.objectTrackingState != sl.OBJECT_TRACKING_STATE.OK) || float.IsInfinity(obj.position.X))
                continue;

            bool new_object = true;
            foreach (Tracklet track in tracklets)
            {
                if (track.id == id && track.is_alive)
                {
                    new_object = false;
                    track.addDetectedPoint(obj, current_timestamp, smoothing_window_size);
                }
            }

            // In case this object does not belong to existing tracks
            if (new_object)
            {
                Tracklet new_track = new Tracklet(obj, obj.label, current_timestamp);
                tracklets.Add(new_track);
            }
        }
    }

    void detectUnchangedTrack(ulong current_timestamp){
        for (int track_index = 0; track_index < tracklets.Count; ++track_index)
        {
            Tracklet track = tracklets[track_index];
            if (track.last_detected_timestamp < current_timestamp && track.last_detected_timestamp > 0)
            {
                // If track missed more than N frames, delete it
                if (current_timestamp - track.last_detected_timestamp >= history_duration)
                {
                    track.is_alive = false;
                    continue;
                }
            }
        }
    }

    void pruneOldPoints(ulong ts){
        List<int> track_to_delete = new List<int>(); // If a dead track does not contain drawing points, juste erase it
        for (int track_index = 0; track_index < tracklets.Count; ++track_index)
        {
            if (tracklets[track_index].is_alive)
            {
                while (tracklets[track_index].positions.Count > 0 && tracklets[track_index].positions[0].timestamp < ts - history_duration)
                {
                    tracklets[track_index].positions.RemoveAt(0);
                }
                while (tracklets[track_index].positions_to_draw.Count > 0 && tracklets[track_index].positions_to_draw[0].timestamp < ts - history_duration)
                {
                    tracklets[track_index].positions_to_draw.RemoveAt(0);
                }
            }
            else
            { // Here, we fade the dead trajectories faster than the alive one (4 points every frame)
                for (int i = 0; i < 4; ++i)
                {
                    if (tracklets[track_index].positions.Count > 0)
                    {
                        tracklets[track_index].positions.RemoveAt(0);
                    }
                    if (tracklets[track_index].positions_to_draw.Count > 0)
                    {
                        tracklets[track_index].positions_to_draw.RemoveAt(0);
                    }
                    else
                    {
                        track_to_delete.Add(track_index);
                        break;
                    }
                }
            }
        }

        int size_ = (int)(track_to_delete.Count - 1);
        for (int i = size_; i >= 0; --i)
            tracklets.RemoveAt(track_to_delete[i]);
    }

    void computeFOV(){
        sl.Resolution image_size = camera_calibration.leftCam.resolution;
        float fx = camera_calibration.leftCam.fx;
        fov = 2.0f * (float)Math.Atan((int)image_size.width / (2.0f * fx));
    }

    void zoom(float factor)
    {
        x_min *= factor;
        x_max *= factor;
        z_min *= factor;

        // Recompute x_step and z_step
        x_step = (x_max - x_min) / window_width;
        z_step = Math.Abs(z_min) / (window_height - camera_offset);
    }

    // background generation
    void generateBackground(){
        // Draw camera + hotkeys information
        drawCamera();
        drawHotkeys();
        has_background_ready = true;
    }

    void drawCamera(){
        // Configuration
        Scalar camera_color  = new Scalar(255, 117, 44, 255);

        int camera_size = 10;
        int camera_height = window_height - camera_offset;
        Point camera_left_pt = new Point(window_width / 2 - camera_size / 2, camera_height);
        Point camera_right_pt = new Point(window_width / 2 + camera_size / 2, camera_height);

        // Drawing camera
        List<Point> camera_pts = new List<Point>{
            new Point(window_width / 2 - camera_size, camera_height),
            new Point(window_width / 2 + camera_size, camera_height),
            new Point(window_width / 2 + camera_size, camera_height + camera_size / 2),
            new Point(window_width / 2 - camera_size, camera_height + camera_size / 2)};

        Cv2.FillConvexPoly(background, camera_pts, camera_color);

        // Compute the FOV
        if (fov < 0.0f)
            computeFOV();
        
        // Get FOV intersection with window borders
        float z_at_x_max = x_max / (float)Math.Tan(fov / 2.0f);
        Point left_intersection_pt = toCVPoint(x_min, -z_at_x_max), right_intersection_pt = toCVPoint(x_max, -z_at_x_max);

        Scalar clr = camera_color;
        // Draw FOV
        // Second try: dotted line
        LineIterator left_line_it = new LineIterator(background, camera_left_pt, left_intersection_pt, PixelConnectivity.Connectivity8);
        int i = 0;
        foreach (var it in left_line_it)
        {
            Point current_pos = it.Pos;
            /*if (i % 5 == 0 || i % 5 == 1)
            {
                it.SetValue<Scalar>(clr);
            }*/
            
            for (int r = 0; r < current_pos.Y; ++r)
            {
                float ratio =(float)r / camera_height;
                background.At<Vec4b>(r, current_pos.X) = Utils.applyFading(background_color, ratio, fov_color);
            }
            i++;
        }

        LineIterator right_line_it = new LineIterator(background, camera_right_pt, right_intersection_pt, PixelConnectivity.Connectivity8);
        int j = 0;
        foreach (var it in right_line_it)
        {
            Point current_pos = it.Pos;
            /*if (j % 5 == 0 || j % 5 == 1)
            {
                it.SetValue<Scalar>(clr);
            }*/

            for (int r = 0; r < current_pos.Y; ++r)
            {
                float ratio = (float)r / camera_height;
                background.At<Vec4b>(r, current_pos.X) = Utils.applyFading(background_color, ratio, fov_color);
            }
            j++;
        }

        for (int c = window_width / 2 - camera_size / 2; c <= window_width / 2 + camera_size / 2; ++c)
        {
            for (int r = 0; r < camera_height; ++r)
            {
                float ratio = (float)r / camera_height;
                background.At<Vec4b>(r, c) = Utils.applyFading(background_color, ratio, fov_color);
            }
        }
    }

    void drawHotkeys(){
        Scalar hotkeys_clr = new Scalar(0, 0, 0, 255);
        Cv2.PutText(background, "Press 'i' to zoom in", new Point(25, window_height - 25), HersheyFonts.HersheyPlain,
                1.0, hotkeys_clr, 1);
       Cv2.PutText(background, "Press 'o' to zoom out", new Point(25, window_height - 15), HersheyFonts.HersheyPlain,
                1.0, hotkeys_clr, 1);
    }

    // Utils
    Point toCVPoint(double x, double z)
    {
        return new Point((x - x_min) / x_step, (z - z_min) / z_step);
    }

    Point toCVPoint(Vector3 position, sl.Pose pose)
    {
        // Go to camera current pose
        Quaternion rotation = pose.rotation;
        Quaternion rotation_inv = Quaternion.Inverse(rotation);
        Vector3 new_position = Vector3.Transform(position - pose.translation, rotation_inv);
        return new Point((int)((new_position.X - x_min) / x_step + .5f), (int)((new_position.Z - z_min) / z_step + .5f));
    }

    Point toCVPoint(TrackPoint position, sl.Pose pose)
    {
        Vector3 sl_position = position.toVector3();
        return toCVPoint(sl_position, pose);
    }
}
