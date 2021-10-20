//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using sl;

using OpenCvSharp;


public class TrackingViewer
{

    private static readonly Tuple<sl.BODY_PARTS, sl.BODY_PARTS>[] SKELETON_BONES =
    {
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.NECK),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NECK, BODY_PARTS.RIGHT_SHOULDER),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_SHOULDER, BODY_PARTS.RIGHT_ELBOW),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_ELBOW, BODY_PARTS.RIGHT_WRIST),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NECK, BODY_PARTS.LEFT_SHOULDER),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_SHOULDER, BODY_PARTS.LEFT_ELBOW),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_ELBOW, BODY_PARTS.LEFT_WRIST),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_HIP, BODY_PARTS.RIGHT_KNEE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_KNEE, BODY_PARTS.RIGHT_ANKLE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_HIP, BODY_PARTS.LEFT_KNEE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_KNEE, BODY_PARTS.LEFT_ANKLE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_SHOULDER, BODY_PARTS.LEFT_SHOULDER),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_HIP, BODY_PARTS.LEFT_HIP),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.RIGHT_EYE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_EYE, BODY_PARTS.RIGHT_EAR),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.LEFT_EYE),
        Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_EYE, BODY_PARTS.LEFT_EAR)
    };

    private static readonly Tuple<sl.BODY_PARTS_POSE_34, sl.BODY_PARTS_POSE_34>[] BODY_BONES_POSE_34 =
{
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.PELVIS, BODY_PARTS_POSE_34.NAVAL_SPINE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.NAVAL_SPINE, BODY_PARTS_POSE_34.CHEST_SPINE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.CHEST_SPINE, BODY_PARTS_POSE_34.LEFT_CLAVICLE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_CLAVICLE, BODY_PARTS_POSE_34.LEFT_SHOULDER),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_SHOULDER, BODY_PARTS_POSE_34.LEFT_ELBOW),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_ELBOW, BODY_PARTS_POSE_34.LEFT_WRIST),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_WRIST, BODY_PARTS_POSE_34.LEFT_HAND),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_HAND, BODY_PARTS_POSE_34.LEFT_HANDTIP),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_WRIST, BODY_PARTS_POSE_34.LEFT_THUMB),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.CHEST_SPINE, BODY_PARTS_POSE_34.RIGHT_CLAVICLE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_CLAVICLE, BODY_PARTS_POSE_34.RIGHT_SHOULDER),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_SHOULDER, BODY_PARTS_POSE_34.RIGHT_ELBOW),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_ELBOW, BODY_PARTS_POSE_34.RIGHT_WRIST),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_WRIST, BODY_PARTS_POSE_34.RIGHT_HAND),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_HAND, BODY_PARTS_POSE_34.RIGHT_HANDTIP),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_WRIST, BODY_PARTS_POSE_34.RIGHT_THUMB),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.PELVIS, BODY_PARTS_POSE_34.LEFT_HIP),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_HIP, BODY_PARTS_POSE_34.LEFT_KNEE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_KNEE, BODY_PARTS_POSE_34.LEFT_ANKLE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_ANKLE, BODY_PARTS_POSE_34.LEFT_FOOT),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.PELVIS, BODY_PARTS_POSE_34.RIGHT_HIP),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_HIP, BODY_PARTS_POSE_34.RIGHT_KNEE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_KNEE, BODY_PARTS_POSE_34.RIGHT_ANKLE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_ANKLE, BODY_PARTS_POSE_34.RIGHT_FOOT),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.CHEST_SPINE, BODY_PARTS_POSE_34.NECK),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.NECK, BODY_PARTS_POSE_34.HEAD),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.HEAD, BODY_PARTS_POSE_34.NOSE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.NOSE, BODY_PARTS_POSE_34.LEFT_EYE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_EYE, BODY_PARTS_POSE_34.LEFT_EAR),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.NOSE, BODY_PARTS_POSE_34.RIGHT_EYE),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_EYE, BODY_PARTS_POSE_34.RIGHT_EAR),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_ANKLE, BODY_PARTS_POSE_34.LEFT_HEEL),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_ANKLE, BODY_PARTS_POSE_34.RIGHT_HEEL),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.LEFT_HEEL, BODY_PARTS_POSE_34.LEFT_FOOT),
        Tuple.Create<BODY_PARTS_POSE_34, BODY_PARTS_POSE_34>(BODY_PARTS_POSE_34.RIGHT_HEEL, BODY_PARTS_POSE_34.RIGHT_FOOT)
        };

    static float[,] id_colors = new float[8, 3]{
        { 232.0f, 176.0f ,59.0f },
        { 165.0f, 218.0f ,25.0f },
        { 102.0f, 205.0f ,105.0f},
        { 185.0f, 0.0f   ,255.0f},
        { 99.0f, 107.0f  ,252.0f},
        {252.0f, 225.0f, 8.0f},
        {167.0f, 130.0f, 141.0f},
        {194.0f, 72.0f, 113.0f}
    };

    static Scalar generateColorID(int idx)
    {
        Scalar default_color = new Scalar(236, 184, 36, 255);

        if (idx < 0) return default_color;

        int offset = Math.Max(0, idx % 8);
        Scalar color = new Scalar();
        color[0]= id_colors[offset, 0];
        color[1] = id_colors[offset, 1];
        color[2] = id_colors[offset, 2];
        color[3] = 255.0f;
        return color;
    }

    static Point cvt(Vector2 point, sl.float2 scale)
    {
        return new Point(point.X * scale.x, point.Y * scale.y);
    }

    static bool renderObject(ObjectData i, bool showOnlyOK)
    {
        if (showOnlyOK)
            return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK);
        else
            return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK || i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OFF);
    }

    static sl.float2 getImagePosition(Vector2[] bounding_box_image, sl.float2 img_scale)
    {
        sl.float2 position;
        position.x = (bounding_box_image[0].X + (bounding_box_image[2].X - bounding_box_image[0].X) * 0.5f) * img_scale.x;
        position.y = (bounding_box_image[0].Y + (bounding_box_image[2].Y - bounding_box_image[0].Y) * 0.5f) * img_scale.y;
        return position;
    }

    public static void render_2D(ref OpenCvSharp.Mat left_display, sl.float2 img_scale, ref sl.Objects objects, bool showOnlyOK, sl.BODY_FORMAT body_format)
    {
        OpenCvSharp.Mat overlay = left_display.Clone();
        OpenCvSharp.Rect roi_render = new OpenCvSharp.Rect(1, 1, left_display.Size().Width, left_display.Size().Height);

        for (int i = 0; i < objects.numObject; i++)
        {
            sl.ObjectData obj = objects.objectData[i];
            if (renderObject(obj, showOnlyOK))
            {
                // Draw Skeleton bones
                OpenCvSharp.Scalar base_color = generateColorID(obj.id);
                if (body_format == BODY_FORMAT.POSE_18)
                {
                    foreach (var part in SKELETON_BONES)
                    {
                        var kp_a = cvt(obj.keypoints2D[(int)part.Item1], img_scale);
                        var kp_b = cvt(obj.keypoints2D[(int)part.Item2], img_scale);
                        if (roi_render.Contains(kp_a) && roi_render.Contains(kp_b))
                        {
                            Cv2.Line(left_display, kp_a, kp_b, base_color, 1, LineTypes.AntiAlias);
                        }
                    }

                    var hip_left = obj.keypoints2D[(int)sl.BODY_PARTS.LEFT_HIP];
                    var hip_right = obj.keypoints2D[(int)sl.BODY_PARTS.RIGHT_HIP];
                    var spine = (hip_left + hip_right) / 2;
                    var neck = obj.keypoints2D[(int)sl.BODY_PARTS.NECK];

                    if (hip_left.X > 0 && hip_left.Y > 0 && hip_right.X > 0 && hip_right.Y > 0 && neck.X > 0 && neck.Y > 0)
                    {
                        var spine_a = cvt(spine, img_scale);
                        var spine_b = cvt(neck, img_scale);
                        if (roi_render.Contains(spine_a) && roi_render.Contains(spine_b))
                        {
                            Cv2.Line(left_display, spine_a, spine_b, base_color, 1, LineTypes.AntiAlias);
                        }
                    }

                    // Draw Skeleton joints
                    foreach (var kp in obj.keypoints2D)
                    {
                        Point cv_kp = cvt(kp, img_scale);
                        if (roi_render.Contains(cv_kp))
                        {
                            Cv2.Circle(left_display, cv_kp, 3, base_color, -1);
                        }
                    }

                    if (hip_left.X > 0 && hip_left.Y > 0 && hip_right.X > 0 && hip_right.Y > 0)
                    {
                        Point cv_spine = cvt(spine, img_scale);
                        if (roi_render.Contains(cv_spine))
                        {
                            Cv2.Circle(left_display, cv_spine, 3, base_color, -1);
                        }
                    }
                }
                else if (body_format == BODY_FORMAT.POSE_34){

                    foreach (var part in BODY_BONES_POSE_34)
                    {
                        var kp_a = cvt(obj.keypoints2D[(int)part.Item1], img_scale);
                        var kp_b = cvt(obj.keypoints2D[(int)part.Item2], img_scale);
                        if (roi_render.Contains(kp_a) && roi_render.Contains(kp_b))
                        {
                            Cv2.Line(left_display, kp_a, kp_b, base_color, 1, LineTypes.AntiAlias);
                        }
                    }

                    // Draw Skeleton joints
                    foreach (var kp in obj.keypoints2D)
                    {
                        Point cv_kp = cvt(kp, img_scale);
                        if (roi_render.Contains(cv_kp))
                        {
                            Cv2.Circle(left_display, cv_kp, 3, base_color, -1);
                        }
                    }
                }
            }
        }

        // Here, overlay is as the left image, but with opaque masks on each detected objects
        Cv2.AddWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
    }
}
