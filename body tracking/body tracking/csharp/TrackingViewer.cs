//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using sl;

using OpenCvSharp;


public class TrackingViewer
{

    private static readonly Tuple<sl.BODY_38_PARTS, sl.BODY_38_PARTS>[] SKELETON_BONES_POSE_38 =
    {
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.PELVIS, BODY_38_PARTS.SPINE_1),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.SPINE_1, BODY_38_PARTS.SPINE_2),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.SPINE_2, BODY_38_PARTS.SPINE_3),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.SPINE_3, BODY_38_PARTS.NECK),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.PELVIS, BODY_38_PARTS.LEFT_HIP),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.PELVIS, BODY_38_PARTS.RIGHT_HIP),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.NECK, BODY_38_PARTS.NOSE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.NECK, BODY_38_PARTS.LEFT_CLAVICLE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_CLAVICLE, BODY_38_PARTS.LEFT_SHOULDER),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.NECK, BODY_38_PARTS.RIGHT_CLAVICLE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_CLAVICLE, BODY_38_PARTS.RIGHT_SHOULDER),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.NOSE, BODY_38_PARTS.LEFT_EYE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_EYE, BODY_38_PARTS.LEFT_EAR),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.NOSE, BODY_38_PARTS.RIGHT_EYE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_EYE, BODY_38_PARTS.RIGHT_EAR),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_SHOULDER, BODY_38_PARTS.LEFT_ELBOW),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_ELBOW, BODY_38_PARTS.LEFT_WRIST),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_THUMB_4),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_INDEX_1),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_MIDDLE_4),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_PINKY_1),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_SHOULDER, BODY_38_PARTS.RIGHT_ELBOW),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_ELBOW, BODY_38_PARTS.RIGHT_WRIST),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_THUMB_4),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_INDEX_1),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_MIDDLE_4),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_PINKY_1),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_HIP, BODY_38_PARTS.LEFT_KNEE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_KNEE, BODY_38_PARTS.LEFT_ANKLE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_HEEL),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_BIG_TOE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_SMALL_TOE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_HIP, BODY_38_PARTS.RIGHT_KNEE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_KNEE, BODY_38_PARTS.RIGHT_ANKLE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_HEEL),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_BIG_TOE),
        Tuple.Create<BODY_38_PARTS, BODY_38_PARTS>(BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_SMALL_TOE)
    };

    private static readonly Tuple<sl.BODY_70_PARTS, sl.BODY_70_PARTS>[] BODY_BONES_POSE_70 =
    {
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.PELVIS, BODY_70_PARTS.SPINE_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.SPINE_1, BODY_70_PARTS.SPINE_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.SPINE_2, BODY_70_PARTS.SPINE_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.SPINE_3, BODY_70_PARTS.NECK),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.PELVIS, BODY_70_PARTS.LEFT_HIP),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.PELVIS, BODY_70_PARTS.RIGHT_HIP),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.NECK, BODY_70_PARTS.NOSE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.NECK, BODY_70_PARTS.LEFT_CLAVICLE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_CLAVICLE, BODY_70_PARTS.LEFT_SHOULDER),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.NECK, BODY_70_PARTS.RIGHT_CLAVICLE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_CLAVICLE, BODY_70_PARTS.RIGHT_SHOULDER),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.NOSE, BODY_70_PARTS.LEFT_EYE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_EYE, BODY_70_PARTS.LEFT_EAR),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.NOSE, BODY_70_PARTS.RIGHT_EYE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_EYE, BODY_70_PARTS.RIGHT_EAR),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_SHOULDER, BODY_70_PARTS.LEFT_ELBOW),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_ELBOW, BODY_70_PARTS.LEFT_WRIST),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_WRIST, BODY_70_PARTS.LEFT_HAND_THUMB_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_THUMB_1, BODY_70_PARTS.LEFT_HAND_THUMB_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_THUMB_2, BODY_70_PARTS.LEFT_HAND_THUMB_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_THUMB_3, BODY_70_PARTS.LEFT_HAND_THUMB_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_WRIST, BODY_70_PARTS.LEFT_HAND_INDEX_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_INDEX_1, BODY_70_PARTS.LEFT_HAND_INDEX_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_INDEX_2, BODY_70_PARTS.LEFT_HAND_INDEX_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_INDEX_3, BODY_70_PARTS.LEFT_HAND_INDEX_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_WRIST, BODY_70_PARTS.LEFT_HAND_MIDDLE_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_MIDDLE_1, BODY_70_PARTS.LEFT_HAND_MIDDLE_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_MIDDLE_2, BODY_70_PARTS.LEFT_HAND_MIDDLE_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_MIDDLE_3, BODY_70_PARTS.LEFT_HAND_MIDDLE_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_WRIST, BODY_70_PARTS.LEFT_HAND_RING_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_RING_1, BODY_70_PARTS.LEFT_HAND_RING_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_RING_2, BODY_70_PARTS.LEFT_HAND_RING_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_RING_3, BODY_70_PARTS.LEFT_HAND_RING_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_WRIST, BODY_70_PARTS.LEFT_HAND_PINKY_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_PINKY_1, BODY_70_PARTS.LEFT_HAND_PINKY_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_PINKY_2, BODY_70_PARTS.LEFT_HAND_PINKY_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HAND_PINKY_3, BODY_70_PARTS.LEFT_HAND_PINKY_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_SHOULDER, BODY_70_PARTS.RIGHT_ELBOW),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_ELBOW, BODY_70_PARTS.RIGHT_WRIST),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_WRIST, BODY_70_PARTS.RIGHT_HAND_THUMB_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_THUMB_1, BODY_70_PARTS.RIGHT_HAND_THUMB_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_THUMB_2, BODY_70_PARTS.RIGHT_HAND_THUMB_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_THUMB_3, BODY_70_PARTS.RIGHT_HAND_THUMB_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_WRIST, BODY_70_PARTS.RIGHT_HAND_INDEX_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_INDEX_1, BODY_70_PARTS.RIGHT_HAND_INDEX_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_INDEX_2, BODY_70_PARTS.RIGHT_HAND_INDEX_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_INDEX_3, BODY_70_PARTS.RIGHT_HAND_INDEX_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_WRIST, BODY_70_PARTS.RIGHT_HAND_MIDDLE_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_MIDDLE_1, BODY_70_PARTS.RIGHT_HAND_MIDDLE_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_MIDDLE_2, BODY_70_PARTS.RIGHT_HAND_MIDDLE_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_MIDDLE_3, BODY_70_PARTS.RIGHT_HAND_MIDDLE_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_WRIST, BODY_70_PARTS.RIGHT_HAND_RING_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_RING_1, BODY_70_PARTS.RIGHT_HAND_RING_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_RING_2, BODY_70_PARTS.RIGHT_HAND_RING_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_RING_3, BODY_70_PARTS.RIGHT_HAND_RING_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_WRIST, BODY_70_PARTS.RIGHT_HAND_PINKY_1),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_PINKY_1, BODY_70_PARTS.RIGHT_HAND_PINKY_2),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_PINKY_2, BODY_70_PARTS.RIGHT_HAND_PINKY_3),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HAND_PINKY_3, BODY_70_PARTS.RIGHT_HAND_PINKY_4),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_HIP, BODY_70_PARTS.LEFT_KNEE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_ANKLE, BODY_70_PARTS.LEFT_HEEL),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_ANKLE, BODY_70_PARTS.LEFT_BIG_TOE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.LEFT_ANKLE, BODY_70_PARTS.LEFT_SMALL_TOE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_HIP, BODY_70_PARTS.RIGHT_KNEE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_ANKLE, BODY_70_PARTS.RIGHT_HEEL),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_ANKLE, BODY_70_PARTS.RIGHT_BIG_TOE),
        Tuple.Create<BODY_70_PARTS, BODY_70_PARTS>(BODY_70_PARTS.RIGHT_ANKLE, BODY_70_PARTS.RIGHT_SMALL_TOE)
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

    static bool renderBody(BodyData i, bool showOnlyOK)
    {
        if (showOnlyOK)
            return (i.trackingState == sl.OBJECT_TRACKING_STATE.OK);
        else
            return (i.trackingState == sl.OBJECT_TRACKING_STATE.OK || i.trackingState == sl.OBJECT_TRACKING_STATE.OFF);
    }

    static sl.float2 getImagePosition(Vector2[] bounding_box_image, sl.float2 img_scale)
    {
        sl.float2 position;
        position.x = (bounding_box_image[0].X + (bounding_box_image[2].X - bounding_box_image[0].X) * 0.5f) * img_scale.x;
        position.y = (bounding_box_image[0].Y + (bounding_box_image[2].Y - bounding_box_image[0].Y) * 0.5f) * img_scale.y;
        return position;
    }

    public static void render_2D(ref OpenCvSharp.Mat left_display, sl.float2 img_scale, ref sl.Bodies bodies, bool showOnlyOK, sl.BODY_FORMAT body_format)
    {
        OpenCvSharp.Mat overlay = left_display.Clone();
        OpenCvSharp.Rect roi_render = new OpenCvSharp.Rect(1, 1, left_display.Size().Width, left_display.Size().Height);

        for (int i = 0; i < bodies.nbBodies; i++)
        {
            sl.BodyData bod = bodies.bodiesList[i];
            if (renderBody(bod, showOnlyOK))
            {
                // Draw Skeleton bones
                OpenCvSharp.Scalar base_color = generateColorID(bod.id);
                if (body_format == BODY_FORMAT.POSE_38)
                {
                    foreach (var part in SKELETON_BONES_POSE_38)
                    {
                        var kp_a = cvt(bod.keypoints2D[(int)part.Item1], img_scale);
                        var kp_b = cvt(bod.keypoints2D[(int)part.Item2], img_scale);
                        if (roi_render.Contains(kp_a) && roi_render.Contains(kp_b))
                        {
                            Cv2.Line(left_display, kp_a, kp_b, base_color, 1, LineTypes.AntiAlias);
                        }
                    }

                    // Draw Skeleton joints
                    foreach (var kp in bod.keypoints2D)
                    {
                        Point cv_kp = cvt(kp, img_scale);
                        if (roi_render.Contains(cv_kp))
                        {
                            Cv2.Circle(left_display, cv_kp, 3, base_color, -1);
                            //Cv2.PutText(left_display, Array.IndexOf(bod.keypoints2D,kp).ToString(), cv_kp, HersheyFonts.HersheyPlain, 1, base_color);
                        }
                    }
                }
                else if (body_format == BODY_FORMAT.POSE_70){

                    foreach (var part in BODY_BONES_POSE_70)
                    {
                        var kp_a = cvt(bod.keypoints2D[(int)part.Item1], img_scale);
                        var kp_b = cvt(bod.keypoints2D[(int)part.Item2], img_scale);
                        if (roi_render.Contains(kp_a) && roi_render.Contains(kp_b))
                        {
                            Cv2.Line(left_display, kp_a, kp_b, base_color, 1, LineTypes.AntiAlias);
                        }
                    }

                    // Draw Skeleton joints
                    foreach (var kp in bod.keypoints2D)
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

        // Here, overlay is as the left image, but with opaque masks on each detected body
        Cv2.AddWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
    }
}
