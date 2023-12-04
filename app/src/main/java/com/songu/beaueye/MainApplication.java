package com.songu.beaueye;

import android.app.Application;
import android.util.Log;

import com.songu.beaueye.doc.Globals;
import com.songu.beaueye.engine.YoloSegment;

public class MainApplication extends Application {

    public void onCreate() {
        super.onCreate();
//        Globals.yoloSegment = new YoloSegment();
//        new Thread() {
//            public void run() {
//                boolean ret_init = Globals.yoloSegment.Init(getAssets());
//                Log.e("MainActivity", String.valueOf(ret_init));
//            }
//        }.start();
    }
}
