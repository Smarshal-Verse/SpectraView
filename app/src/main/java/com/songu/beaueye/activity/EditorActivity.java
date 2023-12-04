package com.songu.beaueye.activity;

import android.app.Activity;
import android.app.ProgressDialog;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.jsibbold.zoomage.ZoomageView;
import com.songu.beaueye.R;
import com.songu.beaueye.doc.Globals;
import com.songu.beaueye.engine.YoloSegment;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class EditorActivity extends Activity implements View.OnClickListener {
    private ZoomageView imgPick, imgOrg;
    private LinearLayout llGap, llThickness, llColor, llPickColor;
    private ImageView imgGap, imgThickness, imgColor, imgPickColor1, imgPickColor2, imgPickColor3;
    private TextView txtGap, txtThinkness, txtColor, txtGapValue, txtThicknessValue, txtColorValue;
    private SeekBar seekValue;
    private int currentMode = -1;

    private int gapValue = 1;
    private int thinkness = 1;

    private int colorValue = 0;

    public ProgressDialog progressDialog;

    private Bitmap bitmapDrawing = null;
    private YoloSegment yoloSegment = new YoloSegment();
    private boolean isDetectEye = false;

    private TextView txtOriginal, txtSave;
    private ImageView imgBack;

    private boolean ret_init = false;


    public Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == 0) {
                progressDialog.dismiss();
            }
        }
    };

    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_editor);

        initView();
    }

    public void initView() {
        imgPick = (ZoomageView) findViewById(R.id.imgPickedImage);
        imgOrg = (ZoomageView) findViewById(R.id.imgPickedOrgImage);
        imgGap = (ImageView) findViewById(R.id.imgSpace);
        imgThickness = (ImageView) findViewById(R.id.imgThickness);
        imgColor = (ImageView) findViewById(R.id.imgColor);
        llGap = (LinearLayout) findViewById(R.id.llGap);
        llThickness = (LinearLayout) findViewById(R.id.llThickness);
        llPickColor = (LinearLayout) findViewById(R.id.llPickColor);
        imgPickColor1 = (ImageView) findViewById(R.id.imgPickColor1);
        imgPickColor2 = (ImageView) findViewById(R.id.imgPickColor2);
        imgPickColor3 = (ImageView) findViewById(R.id.imgPickColor3);
        seekValue = (SeekBar) findViewById(R.id.seekValue);
        llColor = (LinearLayout) findViewById(R.id.llColor);
        txtColor = (TextView) findViewById(R.id.txtColor);
        txtGap = (TextView) findViewById(R.id.txtGap);
        txtThinkness = (TextView) findViewById(R.id.txtThickness);
        txtGapValue = (TextView) findViewById(R.id.txtGapValue);
        txtThicknessValue = (TextView) findViewById(R.id.txtThicknessValue);
        txtColorValue = (TextView) findViewById(R.id.txtColorValue);
        txtOriginal = (TextView) findViewById(R.id.txtOriginalImage);
        txtSave = (TextView) findViewById(R.id.txtPhotoSave);
        imgBack = (ImageView) findViewById(R.id.imgBack);

        progressDialog = new ProgressDialog(this);
        progressDialog.setCancelable(false);
        progressDialog.setTitle("Loading engine");
        progressDialog.setIndeterminate(true);

        seekValue.setVisibility(View.GONE);
        llPickColor.setVisibility(View.GONE);

        llGap.setOnClickListener(EditorActivity.this);
        llThickness.setOnClickListener(EditorActivity.this);
        llColor.setOnClickListener(EditorActivity.this);


        imgPickColor1.setOnClickListener(this);
        imgPickColor2.setOnClickListener(this);
        imgPickColor3.setOnClickListener(this);
        txtSave.setOnClickListener(this);
        txtOriginal.setOnClickListener(this);

        imgBack.setOnClickListener(this);
        if (Globals.pickedImage != null) {
            imgPick.setImageBitmap(Globals.pickedImage);
            imgOrg.setImageBitmap(Globals.pickedImage);
        } else {
            imgPick.setImageURI(Globals.pickedImageGallery);
            imgPick.setRotation(Globals.pickedImageRotated);

            imgOrg.setImageURI(Globals.pickedImageGallery);
            imgOrg.setRotation(Globals.pickedImageRotated);
        }
//        progressDialog.show();

        new Thread() {
            public void run() {
                ret_init = yoloSegment.Init(getAssets());
//                progressDialog.dismiss();

            }
        }.start();
        //mHandler.sendEmptyMessageDelayed(0, 2500);
        seekValue.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Log.e("Value", String.valueOf(i));
                if (currentMode == 0) {
                    gapValue = i;
                    txtGapValue.setText(String.valueOf(i));
                } else if (currentMode == 1) {
                    thinkness = i;
                    txtThicknessValue.setText(String.valueOf(i));
                }
                process();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
    }


//    @Override
//    public boolean onTouchEvent(MotionEvent motionEvent) {
//        scaleGestureDetector.onTouchEvent(motionEvent);
//        return true;
//    }

    public void setController(int mode) {
        currentMode = mode;
        llGap.setBackground(getResources().getDrawable(R.drawable.dw_stroke_white));
        llThickness.setBackground(getResources().getDrawable(R.drawable.dw_stroke_white));
        llColor.setBackground(getResources().getDrawable(R.drawable.dw_stroke_white));

        imgGap.setVisibility(View.VISIBLE);
        imgThickness.setVisibility(View.VISIBLE);
        imgColor.setVisibility(View.VISIBLE);

        txtGapValue.setVisibility(View.GONE);
        txtThicknessValue.setVisibility(View.GONE);
        txtColorValue.setVisibility(View.GONE);

        seekValue.setVisibility(View.GONE);
        llPickColor.setVisibility(View.GONE);

        txtColor.setTextColor(0xff000000);
        txtThinkness.setTextColor(0xff000000);
        txtGap.setTextColor(0xff000000);

        switch (mode) {
            case 0:
                seekValue.setProgress(gapValue);
                seekValue.setMax(3);
                seekValue.setMin(1);
                seekValue.setVisibility(View.VISIBLE);
                txtGap.setTextColor(0xff00C3AD);
                llGap.setBackground(getResources().getDrawable(R.drawable.dw_stroke_green));
                txtGapValue.setVisibility(View.VISIBLE);
                imgGap.setVisibility(View.GONE);
                break;
            case 1:
                seekValue.setProgress(thinkness);
                seekValue.setMax(3);
                seekValue.setMin(1);
                seekValue.setVisibility(View.VISIBLE);
                txtThinkness.setTextColor(0xff00C3AD);
                llThickness.setBackground(getResources().getDrawable(R.drawable.dw_stroke_green));
                txtThicknessValue.setVisibility(View.VISIBLE);
                imgThickness.setVisibility(View.GONE);
                break;
            case 2:
                llPickColor.setVisibility(View.VISIBLE);
                txtColor.setTextColor(0xff00C3AD);
                llColor.setBackground(getResources().getDrawable(R.drawable.dw_stroke_green));
                txtColorValue.setVisibility(View.VISIBLE);
                imgColor.setVisibility(View.GONE);
                selectColor(colorValue);
                break;
        }
    }

    public void selectColor(int index) {
        colorValue = index;
        imgPickColor1.setBackground(getResources().getDrawable(R.drawable.dw_color_2));
        imgPickColor2.setBackground(getResources().getDrawable(R.drawable.dw_color_3));
        imgPickColor3.setBackground(getResources().getDrawable(R.drawable.dw_color_1));

        txtColorValue.setBackground(getResources().getDrawable(R.drawable.dw_color_2));

        switch (index) {
            case 0:
                txtColorValue.setBackground(getResources().getDrawable(R.drawable.dw_color_2));
                imgPickColor1.setBackground(getResources().getDrawable(R.drawable.dw_color_stroke2));
                break;
            case 1:
                txtColorValue.setBackground(getResources().getDrawable(R.drawable.dw_color_3));
                imgPickColor2.setBackground(getResources().getDrawable(R.drawable.dw_color_stroke3));
                break;
            case 2:
                txtColorValue.setBackground(getResources().getDrawable(R.drawable.dw_color_1));
                imgPickColor3.setBackground(getResources().getDrawable(R.drawable.dw_color_stroke1));
                break;
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public void detectEye() {

        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        Bitmap bmp = null;
        if (Globals.pickedImage != null) {
            bmp = Globals.pickedImage;
        } else {
            try {
                bmp = MediaStore.Images.Media.getBitmap(EditorActivity.this.getContentResolver(), Globals.pickedImageGallery);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        isDetectEye = yoloSegment.ProcessBitmap(bmp);
        if (!isDetectEye) {
            Toast.makeText(this, "Detect Eye fail. Please try other image", Toast.LENGTH_SHORT).show();
        }
    }

    public void process() {
//        progressDialog.setTitle("Processing...");
//        progressDialog.show();
        new Thread() {
            public void run() {
//                float gap = gapValue / 200.0f;
                float gap = gapValue;
                int thickness = thinkness;
                Bitmap bmp = null;
                if (Globals.pickedImage != null) {
                    bmp = Globals.pickedImage;
                } else {
                    try {
                        bmp = MediaStore.Images.Media.getBitmap(EditorActivity.this.getContentResolver(), Globals.pickedImageGallery);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                bitmapDrawing = bmp.copy(Bitmap.Config.ARGB_8888, true);
                int color = 0xff00CCFF;
                switch (colorValue) {
                    case 0:
                        color = 0xff21201f;
                        break;
                    case 1:
                        color = 0xff37200e;
                        break;
                    case 2:
                        color = 0xff71855a;
                        break;
                }
                Bitmap new_bitmap = yoloSegment.drawEyeCurve(bitmapDrawing, gap, thickness, color);
                EditorActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        imgPick.setImageBitmap(new_bitmap);
                    }
                });

//                progressDialog.dismiss();
            }
        }.start();
    }

    public void saveImageToExternal() {
        //Create Path to save Image
        String imgName = "BeauEye_" + String.valueOf(System.currentTimeMillis()) + ".jpg";
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES); //Creates app specific folder
        path.mkdirs();
        File imageFile = new File(path, imgName); // Imagename.png
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(imageFile);
            Bitmap bm = ((BitmapDrawable) imgPick.getDrawable()).getBitmap();
            bm.compress(Bitmap.CompressFormat.PNG, 100, out); // Compress Image
            out.flush();
            out.close();
            Toast.makeText(EditorActivity.this, "Image saved", Toast.LENGTH_SHORT).show();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void saveImage() {
        String path = Environment.getExternalStorageDirectory().toString();
        OutputStream fOut = null;
        File file = new File(path, "BeauEye_" + String.valueOf(System.currentTimeMillis()) + ".jpg"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
        try {
            fOut = new FileOutputStream(file);
            Bitmap bm = ((BitmapDrawable) imgPick.getDrawable()).getBitmap();
            bm.compress(Bitmap.CompressFormat.JPEG, 85, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
            fOut.flush(); // Not really required
            fOut.close(); // do not forget to close the stream
            MediaStore.Images.Media.insertImage(getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());
            Toast.makeText(EditorActivity.this, "Image saved", Toast.LENGTH_SHORT).show();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.llGap) {
            if (!ret_init) return;
            setController(0);
            if (!isDetectEye) {
                detectEye();
            }
            process();
        } else if (view.getId() == R.id.llThickness) {
            if (!ret_init) return;
            setController(1);
            if (!isDetectEye) {
                detectEye();
            }
            process();
        } else if (view.getId() == R.id.llColor) {
            if (!ret_init) return;
            setController(2);
            if (!isDetectEye) {
                detectEye();
            }
            process();
        } else if (view.getId() == R.id.imgPickColor1) {
            selectColor(0);
            process();
        } else if (view.getId() == R.id.imgPickColor2) {
            selectColor(1);
            process();
        } else if (view.getId() == R.id.imgPickColor3) {
            selectColor(2);
            process();
        } else if (view.getId() == R.id.txtOriginalImage) {
            if (Globals.pickedImage != null) {
                imgPick.setImageBitmap(Globals.pickedImage);
            } else {
                imgPick.setImageURI(Globals.pickedImageGallery);
                imgPick.setRotation(Globals.pickedImageRotated);
            }
        } else if (view.getId() == R.id.txtPhotoSave) {
            if (Build.BRAND.equals("samsung")) {
                saveImageToExternal();
            } else {
                saveImage();
            }
        } else if (view.getId() == R.id.imgBack) {
            finish();
        }
    }
}
