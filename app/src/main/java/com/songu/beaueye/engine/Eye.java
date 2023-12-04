package com.songu.beaueye.engine;

import android.util.Log;

import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Eye {
    public Point[] LANDMARK_EYELID = null;
    public List<Point> LANDMARK_IRIS = null;
    public RotatedRect ELLIPSE_EYE = null;
    public Rect EYELID_BOX = null;
    public String TAG = "YoloSegmentEYE";

    public boolean isValid(){
        if(LANDMARK_EYELID == null || LANDMARK_EYELID.length == 0 ||
                LANDMARK_IRIS == null || LANDMARK_IRIS.isEmpty() ||
                ELLIPSE_EYE == null || EYELID_BOX == null){
            return false;
        } else {
            return true;
        }
    }
    public List<Point> generateEllipsePoint(float scale){
        if (! isValid()){
            Log.d(TAG, "ELLIPSE eye not yet defined!");
            return null;
        }
        List<Point> ellipsePoints = new ArrayList<>();
        float [] angles = new float[360];
        for(int i = 0; i < angles.length; i ++){
            angles[i] = (float) i;
        }

        double center_x = ELLIPSE_EYE.center.x;
        double center_y = ELLIPSE_EYE.center.y;
        double major_axis = (ELLIPSE_EYE.size.width / 2);
        double minor_axis = (ELLIPSE_EYE.size.height / 2);

//        double rotation_angle_radian = Math.toRadians(ELLIPSE_EYE.angle);
        Log.d(TAG, "Ellipse x: " + center_x + ", y: " + center_y + ", w: " + major_axis + ", h: " + minor_axis + ", angle: " + ELLIPSE_EYE.angle);
        if(major_axis < minor_axis){
            double temp = minor_axis;
            minor_axis = major_axis;
            major_axis = temp;
        }
        major_axis = major_axis + major_axis * scale;
        minor_axis = minor_axis + minor_axis * scale;

        for(int i = 0; i < angles.length ; i ++){
            double angle_radian = Math.toRadians(angles[i]);
            double x = center_x + major_axis * Math.cos(angle_radian)  ;
            double y = center_y + minor_axis * Math.sin(angle_radian)  ;
            ellipsePoints.add(new Point(x, y));
        }

        return ellipsePoints;
    }
    public void getEyelidBox(){
        double min_eyelid_x = 1000;
        double max_eyelid_x = -100;
        double min_eyelid_y = 1000;
        double max_eyelid_y = -100;

        for(int i = 0; i < LANDMARK_EYELID.length; i ++){
            double x = LANDMARK_EYELID[i].x;
            double y = LANDMARK_EYELID[i].y;
            if(min_eyelid_x > x){
                min_eyelid_x = x;
            }
            if(max_eyelid_x < x){
                max_eyelid_x = x;
            }
            if(min_eyelid_y > y){
                min_eyelid_y = y;
            }
            if(max_eyelid_y < y){
                max_eyelid_y = y;
            }
        }
        Log.d(TAG, "min_eyelid_x " + min_eyelid_x);
        Log.d(TAG, "min_eyelid_y " + min_eyelid_y);
        Log.d(TAG, "max_eyelid_x " + max_eyelid_x);
        Log.d(TAG, "max_eyelid_y " + max_eyelid_y);
        if(min_eyelid_x > 0 && min_eyelid_y > 0){
            EYELID_BOX = new Rect(
                    (int) min_eyelid_x,
                    (int) min_eyelid_y,
                    (int) (max_eyelid_x - min_eyelid_x),
                    (int) (max_eyelid_y - min_eyelid_y)
            );
        }
    }

    public Eye(List<Point> eyelid_points, List<Point> iris_points){
        /// Process EYELID ///
        LANDMARK_EYELID = new Point[eyelid_points.size()];
        for(int i = 0; i < eyelid_points.size(); i ++){
            LANDMARK_EYELID[i] = eyelid_points.get(i);
//            Log.d(TAG, "LANDMARK_EYELID (" + eyelid_points.get(i).x + ", " + eyelid_points.get(i).y + " )");
        }
        getEyelidBox();

        /// Process IRIS ///
        LANDMARK_IRIS = new ArrayList<>();
        for(int i = 0; i < iris_points.size(); i ++){
            Point check_point = iris_points.get(i);
            double inside_eyelid = Imgproc.pointPolygonTest(
                    new MatOfPoint2f(LANDMARK_EYELID),
                    check_point,
                    false
            );
            if (inside_eyelid > 0) {
                LANDMARK_IRIS.add(check_point);
            }
        }

        if(! LANDMARK_IRIS.isEmpty()){
            Point[] ellipse_iris_points = new Point[LANDMARK_IRIS.size()];
            Log.d(TAG, "ellipse_iris_points "  + LANDMARK_IRIS.size());
            for(int i = 0; i < LANDMARK_IRIS.size(); i ++){
                ellipse_iris_points[i] = LANDMARK_IRIS.get(i);
            }
            RotatedRect minEllipse = Imgproc.fitEllipse(new MatOfPoint2f(ellipse_iris_points));
            Log.d(TAG, "minEllipse x: " + minEllipse.center.x + " y: " + minEllipse.center.y + " w: " + minEllipse.size.width + " h " + minEllipse.size.height);
            ELLIPSE_EYE = minEllipse;
            LANDMARK_IRIS = iris_points;

        }
    }
}
