//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenCvSharp;

namespace sl
{
    public class Utils
    {
        /// <summary>
        ///  Creates an OpenCV version of a ZED Mat. 
        /// </summary>
        /// <param name="zedmat">Source ZED Mat.</param>
        /// <param name="zedmattype">Type of ZED Mat - data type and channel number.
        /// <returns></returns>
        public static OpenCvSharp.Mat SLMat2CVMat(ref sl.Mat zedmat, MAT_TYPE zedmattype)
        {
            int cvmattype = SLMatType2CVMatType(zedmattype);
            OpenCvSharp.Mat cvmat = new OpenCvSharp.Mat(zedmat.GetHeight(), zedmat.GetWidth(), cvmattype, zedmat.GetPtr());

            return cvmat;
        }

        /// <summary>
        /// Returns the OpenCV type that corresponds to a given ZED Mat type. 
        /// </summary>
        private static int SLMatType2CVMatType(MAT_TYPE zedmattype)
        {
            switch (zedmattype)
            {
                case sl.MAT_TYPE.MAT_32F_C1:
                    return OpenCvSharp.MatType.CV_32FC1;
                case sl.MAT_TYPE.MAT_32F_C2:
                    return OpenCvSharp.MatType.CV_32FC2;
                case sl.MAT_TYPE.MAT_32F_C3:
                    return OpenCvSharp.MatType.CV_32FC3;
                case sl.MAT_TYPE.MAT_32F_C4:
                    return OpenCvSharp.MatType.CV_32FC4;
                case sl.MAT_TYPE.MAT_8U_C1:
                    return OpenCvSharp.MatType.CV_8UC1;
                case sl.MAT_TYPE.MAT_8U_C2:
                    return OpenCvSharp.MatType.CV_8UC2;
                case sl.MAT_TYPE.MAT_8U_C3:
                    return OpenCvSharp.MatType.CV_8UC3;
                case sl.MAT_TYPE.MAT_8U_C4:
                    return OpenCvSharp.MatType.CV_8UC4;
                default:
                    return -1;
            }
        }

        public static ulong getMilliseconds(ulong ts_ns)
        {
            return ts_ns / 1000000;
        }

        public static  void drawVerticalLine(ref OpenCvSharp.Mat left_display, Point start_pt, Point end_pt, Scalar clr, int thickness)
        {
            int n_steps = 7;
            Point pt1, pt4;
            pt1.X = ((n_steps - 1) * start_pt.X + end_pt.X) / n_steps;
            pt1.Y = ((n_steps - 1) * start_pt.Y + end_pt.Y) / n_steps;

            pt4.X = (start_pt.X + (n_steps - 1) * end_pt.X) / n_steps;
            pt4.Y = (start_pt.Y + (n_steps - 1) * end_pt.Y) / n_steps;

            Cv2.Line(left_display, start_pt, pt1, clr, thickness);
            Cv2.Line(left_display, pt4, end_pt, clr, thickness);
        }

        public static Point cvt(Vector2 pt, sl.float2 scale)
        {
            return new Point(pt.X * scale.x, pt.Y * scale.y);
        }

        public static sl.float4 generateColorID(int idx)
        {
            int offset = Math.Max(0, idx % 5);
            sl.float4 color = new float4();
            color.x = id_colors[offset, 0];
            color.y = id_colors[offset, 1];
            color.z = id_colors[offset, 2];
            color.w = 1.0f;
            return color;
        }

        public static OpenCvSharp.Scalar generateColorID_u(int idx)
        {
            int offset = Math.Max(0, idx % 5);
            OpenCvSharp.Scalar color = new OpenCvSharp.Scalar();
            color[0] = id_colors[offset, 2] * 255;
            color[1] = id_colors[offset, 1] * 255;
            color[2] = id_colors[offset, 0] * 255;
            color[3] = 1.0f * 255;
            return color;
        }

        public static float[,] id_colors = new float[5, 3]{

        {.231f, .909f, .69f},
        {.098f, .686f, .816f},
        {.412f, .4f, .804f},
        {1, .725f, .0f},
        {.989f, .388f, .419f}
    };

        public static float[,] class_colors = new float[6, 3]{
        { 44.0f, 117.0f, 255.0f}, // PEOPLE
        { 255.0f, 0.0f, 255.0f}, // VEHICLE
        { 0.0f, 0.0f, 255.0f},
        { 0.0f, 255.0f, 255.0f},
        { 0.0f, 255.0f, 0.0f},
        { 255.0f, 255.0f, 255.0f}
    };

        public static float4 generateColorClass(int idx)
        {
            idx = Math.Min(5, idx);
            sl.float4 color = new float4();
            color.x = class_colors[idx, 0];
            color.y = class_colors[idx, 1];
            color.z = class_colors[idx, 2];
            color.w = 255.0f;
            return color;
        }

        public static OpenCvSharp.Scalar generateColorClass_u(int idx)
        {
            idx = Math.Min(5, idx);
            OpenCvSharp.Scalar color = new OpenCvSharp.Scalar();
            color[0] = class_colors[idx, 0];
            color[0] = class_colors[idx, 1];
            color[0] = class_colors[idx, 2];
            color[0] = 255.0f;
            return color;
        }

        bool renderObject(ObjectData i, bool showOnlyOK)
        {
            if (showOnlyOK)
                return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK);
            else
                return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK || i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OFF);
        }


        public static byte _applyFading(double val, float current_alpha, double current_clr)
        {
            return (byte)(current_alpha * current_clr + (1.0 - current_alpha) * val);
        }

        public static Vec4b applyFading(Scalar val, float current_alpha, Scalar current_clr){
             Vec4b out_ = new Vec4b();
             out_[0] = _applyFading(val[0], current_alpha, current_clr[0]);
             out_[1] = _applyFading(val[1], current_alpha, current_clr[1]);
             out_[2] = _applyFading(val[2], current_alpha, current_clr[2]);
             out_[3] = 255;
             return out_;
        }
    }
}