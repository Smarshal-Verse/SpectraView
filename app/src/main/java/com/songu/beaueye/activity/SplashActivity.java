package com.songu.beaueye.activity;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.CountDownTimer;

import androidx.appcompat.app.AppCompatActivity;

import com.songu.beaueye.R;

public class SplashActivity extends AppCompatActivity {

    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_splash);
        new CountDownTimer(2500, 1000) {

            public void onTick(long millisUntilFinished) {

            }

            public void onFinish() {
                Intent m = new Intent(SplashActivity.this, MainActivity.class);
                m.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP| Intent.FLAG_ACTIVITY_CLEAR_TASK);
                SplashActivity.this.startActivity(m);
                SplashActivity.this.finish();
            }
        }.start();
    }

}
