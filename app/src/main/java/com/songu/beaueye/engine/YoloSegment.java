package com.songu.beaueye.engine;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class YoloSegment {

    private String TAG = "YoloSegment";
    private int LABEL_IRIS = 0;
    private int LABEL_EYELID = 1;
    public Point[] LANDMARK_EYELID = null;
    public List<Point> LANDMARK_IRIS = null;
    public List<Eye> EYES = null;

    public native boolean Init(AssetManager mgr);

    public native float[] Detect(Bitmap bitmap, boolean use_gpu);

    public RotatedRect ELLIPSE_EYE = null;
    public Rect EYELID_BOX = null;

    public boolean ProcessBitmap(Bitmap bitmap) {
        Bitmap yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        float[] boundary_segment = Detect(yourSelectedImage, false);
        postProcess(boundary_segment);
        if (isNull()) {
            return false;
        }
        return true;
    }

    public void reset() {
        EYES = null;
    }

    public boolean isNull() {
        if (EYES == null || EYES.isEmpty()) {
            return true;
        } else {
            return false;
        }
    }

    public void Postprocess(float[] segmentPoints) {
        Log.d(TAG, "get points");

        if (segmentPoints.length == 0) {
            Log.d(TAG, "length point is small " + segmentPoints.length);
            return;
        }
        int num_contour = (int) segmentPoints[0];
        Log.d(TAG, "number of masks detected " + num_contour);
        Log.d(TAG, "segmentPoints length  " + segmentPoints.length);

        int max_eyelid_start = -1;
        int max_eyelid_end = -1;
        int max_iris_start = -1;
        int max_iris_end = -1;
        int before_length = 1 + num_contour * 2;

        for (int i = 0; i < num_contour; i++) {
            int label_id = (int) segmentPoints[1 + i * 2];
            int label_point_length = (int) segmentPoints[1 + i * 2 + 1] * 2;
            Log.d(TAG, "label_point_length " + label_point_length);
            if (label_id == LABEL_EYELID) {
                if (label_point_length > (max_eyelid_end - max_eyelid_start)) {
                    max_eyelid_start = before_length;
                    max_eyelid_end = max_eyelid_start + label_point_length;
                }
            }
            if (label_id == LABEL_IRIS) {
                if (label_point_length > (max_iris_end - max_iris_start)) {
                    max_iris_start = before_length;
                    max_iris_end = max_iris_start + label_point_length;
                }
            }
            before_length = label_point_length + before_length;
        }

        if (max_eyelid_start == -1 || max_eyelid_start == -1) {
            Log.d(TAG, "Could not detect iris and eyelid perfectly!");
            return;
        }

        float min_eyelid_x = 1000;
        float max_eyelid_x = -100;
        float min_eyelid_y = 1000;
        float max_eyelid_y = -100;
        LANDMARK_EYELID = new Point[(max_eyelid_end - max_eyelid_start) / 2];
        for (int i = 0; i < (max_eyelid_end - max_eyelid_start) / 2; i++) {
            float x = segmentPoints[max_eyelid_start + i * 2];
            float y = segmentPoints[max_eyelid_start + i * 2 + 1];
            if (min_eyelid_x > x) {
                min_eyelid_x = x;
            }
            if (max_eyelid_x < x) {
                max_eyelid_x = x;
            }
            if (min_eyelid_y > y) {
                min_eyelid_y = y;
            }
            if (max_eyelid_y < y) {
                max_eyelid_y = y;
            }
//            LANDMARK_EYELID.add(new Point(x, y));
            LANDMARK_EYELID[i] = new Point(x, y);
        }
        if (min_eyelid_x > 0 && min_eyelid_y > 0) {
            EYELID_BOX = new Rect(
                    (int) min_eyelid_x,
                    (int) min_eyelid_y,
                    (int) (max_eyelid_x - min_eyelid_x),
                    (int) (max_eyelid_y - min_eyelid_y)
            );
        }


        LANDMARK_IRIS = new ArrayList<>();
        for (int i = 0; i < (max_iris_end - max_iris_start) / 2; i++) {
            float x = segmentPoints[max_iris_start + i * 2];
            float y = segmentPoints[max_iris_start + i * 2 + 1];
            double inside_eyelid = Imgproc.pointPolygonTest(
                    new MatOfPoint2f(LANDMARK_EYELID),
                    new Point(x, y),
                    false
            );
            if (inside_eyelid > 0) {
                LANDMARK_IRIS.add(new Point(x, y));
            }
        }
        Point[] iris_points = new Point[LANDMARK_IRIS.size()];
        for (int i = 0; i < LANDMARK_IRIS.size(); i++) {
            iris_points[i] = LANDMARK_IRIS.get(i);
        }
//        RotatedRect minRect = Imgproc.minAreaRect(new MatOfPoint2f(iris_points));
        RotatedRect minEllipse = Imgproc.fitEllipse(new MatOfPoint2f(iris_points));
        ELLIPSE_EYE = minEllipse;
    }

    private List<Point> getEachSegmentContours(float[] segmentPoints, int start_idx, int end_idx) {
        List<Point> pointList = new ArrayList<>();
        for (int i = 0; i < (end_idx - start_idx) / 2; i++) {
            float x = segmentPoints[start_idx + i * 2];
            float y = segmentPoints[start_idx + i * 2 + 1];
            pointList.add(new Point(x, y));
        }
        return pointList;
    }

    private Point getCenterOfListPoints(List<Point> pointList) {
        Point center = new Point();
        center.x = 0.0f;
        center.y = 0.0f;
        for (int i = 0; i < pointList.size(); i++) {
            center.x += pointList.get(i).x;
            center.y += pointList.get(i).y;
        }
        center.x /= pointList.size();
        center.y /= pointList.size();
        return center;
    }

    private double distance(Point p1, Point p2) {
        return Math.sqrt(
                Math.pow((p1.x - p2.x), 2) +
                        Math.pow((p1.y - p2.y), 2)
        );
    }

    private List<Eye> getEyeListFromSegmentPoints(
            List<List<Point>> list_all_eyelid,
            List<List<Point>> list_all_iris,
            List<Point> list_all_eyelid_center,
            List<Point> list_all_iris_center
    ) {
        List<Eye> eyeList = new ArrayList<Eye>();
        Log.d(TAG, "list_all_eyelid " + list_all_eyelid.size());
        Log.d(TAG, "list_all_iris " + list_all_iris.size());
        Log.d(TAG, "list_all_eyelid_center " + list_all_eyelid_center.size());
        Log.d(TAG, "list_all_iris_center " + list_all_iris_center.size());

        if (list_all_eyelid.size() >= list_all_iris.size()) {
            Log.d(TAG, "getEyeListFromSegmentPoints case 1");
            for (int iidx = 0; iidx < list_all_iris.size(); iidx++) {
                double min_dist = 10000.0f;
                int min_eid = -1;
                for (int eidx = 0; eidx < list_all_eyelid.size(); eidx++) {
                    double dist = distance(list_all_iris_center.get(iidx), list_all_eyelid_center.get(eidx));
                    if (min_dist > dist) {
                        min_dist = dist;
                        min_eid = eidx;
                    }
                }
                Log.d(TAG, "getEyeListFromSegmentPoints min_eid " + min_eid + " dist " + min_dist);
                Eye eye = new Eye(list_all_eyelid.get(min_eid), list_all_iris.get(iidx));
                eyeList.add(eye);
                list_all_eyelid_center.remove(min_eid);
                list_all_eyelid.remove(min_eid);
            }

        } else {
            Log.d(TAG, "getEyeListFromSegmentPoints case 2");
            for (int eidx = 0; eidx < list_all_eyelid.size(); eidx++) {
                double min_dist = 10000.0f;
                int min_iid = -1;
                for (int iidx = 0; iidx < list_all_iris.size(); iidx++) {
                    double dist = distance(list_all_iris_center.get(iidx), list_all_eyelid_center.get(eidx));
                    if (min_dist < dist) {
                        min_dist = dist;
                        min_iid = eidx;
                    }
                }
                Log.d(TAG, "getEyeListFromSegmentPoints min_iid " + min_iid);
                Eye eye = new Eye(list_all_eyelid.get(min_iid), list_all_iris.get(eidx));
                eyeList.add(eye);
                list_all_iris_center.remove(min_iid);
                list_all_iris.remove(min_iid);
            }
        }

        Log.d(TAG, "Number eye " + eyeList.size());

        return eyeList;
    }

    public void postProcess(float[] segmentPoints) {
        Log.d(TAG, "get points");

        if (segmentPoints.length == 0) {
            Log.d(TAG, "length point is small " + segmentPoints.length);
            return;
        }
        int num_contour = (int) segmentPoints[0];
        Log.d(TAG, "number of masks detected " + num_contour);
        Log.d(TAG, "segmentPoints length  " + segmentPoints.length);

        int start_idx = 1 + num_contour * 2;
        List<List<Point>> list_all_eyelid = new ArrayList<>();
        List<List<Point>> list_all_iris = new ArrayList<>();

        List<Point> list_all_center_eyelid = new ArrayList<>();
        List<Point> list_all_center_iris = new ArrayList<>();

        for (int i = 0; i < num_contour; i++) {
            int label_id = (int) segmentPoints[1 + i * 2];
            Log.d(TAG, "label_id " + label_id);
            int label_point_length = (int) segmentPoints[1 + i * 2 + 1] * 2;
            int end_idx = start_idx + label_point_length;
            Log.d(TAG, "start_idx " + start_idx + " end_idx " + end_idx);

            List<Point> pointList = getEachSegmentContours(segmentPoints, start_idx, end_idx);
            if (pointList.isEmpty()) {

                continue;
            }
            if (label_id == LABEL_EYELID) {
                list_all_eyelid.add(pointList);
                list_all_center_eyelid.add(getCenterOfListPoints(pointList));
            }
            if (label_id == LABEL_IRIS) {
                list_all_iris.add(pointList);
                list_all_center_iris.add(getCenterOfListPoints(pointList));
            }
            start_idx += label_point_length;
        }

        int num_eyelid = list_all_eyelid.size();
        int num_iris = list_all_iris.size();
        boolean miss_segment = (num_eyelid != num_iris);
        Log.d(TAG, "miss_segment  " + miss_segment);

        EYES = getEyeListFromSegmentPoints(
                list_all_eyelid,
                list_all_iris,
                list_all_center_eyelid,
                list_all_center_iris
        );

    }

    private List<Point> generateEllipsePoint(float scale) {
        if (ELLIPSE_EYE == null) {
            Log.d(TAG, "ELLIPSE eye not yet defined!");
            return null;
        }
        List<Point> ellipsePoints = new ArrayList<>();
        float[] angles = new float[360];
        for (int i = 0; i < angles.length; i++) {
            angles[i] = (float) i;
        }

        double center_x = ELLIPSE_EYE.center.x;
        double center_y = ELLIPSE_EYE.center.y;
        double major_axis = (ELLIPSE_EYE.size.width / 2);
        double minor_axis = (ELLIPSE_EYE.size.height / 2);

        double rotation_angle_radian = Math.toRadians(ELLIPSE_EYE.angle);
        Log.d(TAG, "Ellipse x: " + center_x + ", y: " + center_y + ", w: " + major_axis + ", h: " + minor_axis + ", angle: " + ELLIPSE_EYE.angle);
        if (major_axis < minor_axis) {
            double temp = minor_axis;
            minor_axis = major_axis;
            major_axis = temp;
        }
        major_axis = major_axis + major_axis * scale;
        minor_axis = minor_axis + minor_axis * scale;

        for (int i = 0; i < angles.length; i++) {
            double angle_radian = Math.toRadians(angles[i]);
            double x = center_x + major_axis * Math.cos(angle_radian);
            double y = center_y + minor_axis * Math.sin(angle_radian);
            ellipsePoints.add(new Point(x, y));
        }

        return ellipsePoints;
    }

    public Bitmap drawEyeCurve(Bitmap bitmap, float gap, int thickness, int color) {
        if (isNull()) {
            Log.d(TAG, "Not detect eye to draw");
            return bitmap;
        }
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(rgba);


//        Paint eyelid_paint = new Paint();
//        eyelid_paint.setStyle(Paint.Style.STROKE);
//        eyelid_paint.setStrokeWidth(4);
//        eyelid_paint.setColor(Color.rgb(139, 125,  96));
//        Paint iris_paint = new Paint();
//        iris_paint.setStyle(Paint.Style.STROKE);
//        iris_paint.setStrokeWidth(4);
//        iris_paint.setColor(Color.rgb(  0, 152, 255));
//
//        for(int i = 0; i < EYES.size(); i ++){
//            Eye eye = EYES.get(i);
//            if (! eye.isValid()){
//                continue;
//            }
//            for(int j = 0; j  < eye.LANDMARK_EYELID.length; j  ++){
//                Point p = eye.LANDMARK_EYELID[j ];
//                canvas.drawCircle(
//                        (float) p.x,
//                        (float) p.y,
//                        3,
//                        eyelid_paint);
//            }
//
//
//            for(int j  = 0; j  < eye.LANDMARK_IRIS.size(); j  ++){
//                Point p = eye.LANDMARK_IRIS.get(j );
//                canvas.drawCircle(
//                        (float) p.x,
//                        (float) p.y,
//                        3,
//                        iris_paint);
//            }
//        }
//        return  rgba;


        Paint ellipse_paint = new Paint();
        ellipse_paint.setStyle(Paint.Style.STROKE);
        ellipse_paint.setStrokeWidth(1);
        ellipse_paint.setColor(color);


        float scale = 0.07f;

        if (thickness == 1) {
            scale = 0.07f;
        } else if (thickness == 2) {
            scale = 0.11f;
        } else if (thickness == 3) {
            scale = 0.15f;
        }
        gap = scale * 1 / 3;

        for (int i = 0; i < EYES.size(); i++) {
            Eye eye = EYES.get(i);
            if (!eye.isValid()) {
                continue;
            }
//            List<Point> ellipsePoints = eye.generateEllipsePoint(0.0f);
//            for(int j = 0; j < ellipsePoints.size(); j ++){
//                Point p = ellipsePoints.get(j);
//                canvas.drawCircle(
//                        (float) p.x,
//                        (float) p.y,
//                        1,
//                        ellipse_paint);
//            }
            List<Point> inner_ellipsePoints = eye.generateEllipsePoint(gap);
            List<Point> outer_ellipsePoints = eye.generateEllipsePoint(scale);
            Point[] inner_curve = new Point[inner_ellipsePoints.size()];
            for (int j = 0; j < inner_ellipsePoints.size(); j++) {
                inner_curve[j] = inner_ellipsePoints.get(j);
            }
            Point[] outer_curve = new Point[outer_ellipsePoints.size()];
            for (int j = 0; j < outer_ellipsePoints.size(); j++) {
                outer_curve[j] = outer_ellipsePoints.get(j);
            }

            for (int y = eye.EYELID_BOX.y; y < eye.EYELID_BOX.y + eye.EYELID_BOX.height; y++) {
                for (int x = eye.EYELID_BOX.x; x < eye.EYELID_BOX.x + eye.EYELID_BOX.width; x++) {
                    Point p = new Point(x, y);
                    double inside_eyelid = Imgproc.pointPolygonTest(
                            new MatOfPoint2f(eye.LANDMARK_EYELID),
                            p,
                            false
                    );
                    double inside_outer = Imgproc.pointPolygonTest(
                            new MatOfPoint2f(outer_curve),
                            p,
                            false
                    );
                    double inside_inner = Imgproc.pointPolygonTest(
                            new MatOfPoint2f(inner_curve),
                            p,
                            false
                    );
                    if (inside_eyelid >= 0 && inside_outer >= 0 && inside_inner < 0) {
                        canvas.drawCircle(
                                (float) p.x,
                                (float) p.y,
                                1,
                                ellipse_paint);
                    }
                }
            }

        }
        return rgba;
    }

    public Bitmap drawEyeCurves(Bitmap bitmap, float gap, int thickness, int color) {
        if (LANDMARK_EYELID == null || LANDMARK_IRIS == null || LANDMARK_EYELID.length == 0 || LANDMARK_IRIS.size() == 0 || EYELID_BOX == null) {
            Log.d(TAG, "Not detect eye to draw");
            return bitmap;
        }

        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        final int[] colors = new int[]{
                Color.rgb(57, 220, 205),
                Color.rgb(59, 235, 255),
                Color.rgb(7, 193, 255),
                Color.rgb(0, 152, 255),
                Color.rgb(34, 87, 255),
                Color.rgb(72, 85, 121),
                Color.rgb(158, 158, 158),
                Color.rgb(139, 125, 96)
        };

        Canvas canvas = new Canvas(rgba);

//        Paint eyelid_paint = new Paint();
//        eyelid_paint.setStyle(Paint.Style.STROKE);
//        eyelid_paint.setStrokeWidth(4);
//        eyelid_paint.setColor(Color.rgb(139, 125,  96));
//
////        for(int i = 0; i < LANDMARK_EYELID.size(); i ++){
////            Point p = LANDMARK_EYELID.get(i);
//        for(int i = 0; i < LANDMARK_EYELID.length; i ++){
//            Point p = LANDMARK_EYELID[i];
//            canvas.drawCircle(
//                    (float) p.x,
//                    (float) p.y,
//                    3,
//                    eyelid_paint);
//        }
//
//        Paint iris_paint = new Paint();
//        iris_paint.setStyle(Paint.Style.STROKE);
//        iris_paint.setStrokeWidth(4);
//        iris_paint.setColor(Color.rgb(  0, 152, 255));
//        for(int i = 0; i < LANDMARK_IRIS.size(); i ++){
//            Point p = LANDMARK_IRIS.get(i);
////        for(int i = 0; i < LANDMARK_IRIS.length; i ++){
////            Point p = LANDMARK_IRIS[i];
//            canvas.drawCircle(
//                    (float) p.x,
//                    (float) p.y,
//                    3,
//                    iris_paint);
//        }

        Paint ellipse_paint = new Paint();
        ellipse_paint.setStyle(Paint.Style.STROKE);
        ellipse_paint.setStrokeWidth(1);
        ellipse_paint.setColor(color);

        float scale = 0.07f;

        if (thickness == 1) {
            scale = 0.07f;
        } else if (thickness == 2) {
            scale = 0.11f;
        } else if (thickness == 3) {
            scale = 0.15f;
        }
        gap = scale * 1 / 3;

        List<Point> inner_ellipsePoints = generateEllipsePoint(gap);
        List<Point> outer_ellipsePoints = generateEllipsePoint(scale);
        Point[] inner_curve = new Point[inner_ellipsePoints.size()];
        for (int i = 0; i < inner_ellipsePoints.size(); i++) {
            inner_curve[i] = inner_ellipsePoints.get(i);
        }
        Point[] outer_curve = new Point[outer_ellipsePoints.size()];
        for (int i = 0; i < outer_ellipsePoints.size(); i++) {
            outer_curve[i] = outer_ellipsePoints.get(i);
        }
        for (int y = EYELID_BOX.y; y < EYELID_BOX.y + EYELID_BOX.height; y++) {
            for (int x = EYELID_BOX.x; x < EYELID_BOX.x + EYELID_BOX.width; x++) {
                Point p = new Point(x, y);
                double inside_eyelid = Imgproc.pointPolygonTest(
                        new MatOfPoint2f(LANDMARK_EYELID),
                        p,
                        false
                );
                double inside_outer = Imgproc.pointPolygonTest(
                        new MatOfPoint2f(outer_curve),
                        p,
                        false
                );
                double inside_inner = Imgproc.pointPolygonTest(
                        new MatOfPoint2f(inner_curve),
                        p,
                        false
                );
                if (inside_eyelid >= 0 && inside_outer >= 0 && inside_inner < 0) {
                    canvas.drawCircle(
                            (float) p.x,
                            (float) p.y,
                            1,
                            ellipse_paint);
                }
            }
        }

        return rgba;
    }

    static {
        System.loadLibrary("yolosegment");
    }
}
