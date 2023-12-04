package com.songu.beaueye.activity;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.songu.beaueye.R;
import com.songu.beaueye.doc.Globals;
import com.songu.beaueye.util.Utils;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private TextView txtGallery, txtPhoto;
    private static final int CAMERA_REQUEST = 1888;
    private static final int PICK_IMAGE = 1800;


    public Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            grantPermission();
        }
    };

    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_main);
        mHandler.sendEmptyMessageDelayed(0, 400);
        initView();
    }

    public void grantPermission() {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
            } else {
                // Your code
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public void initView() {
        txtGallery = (TextView) this.findViewById(R.id.txtPickGallery);
        txtPhoto = (TextView) this.findViewById(R.id.txtTakePhoto);

        txtGallery.setOnClickListener(this);
        txtPhoto.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.txtPickGallery) {

            Intent i = new Intent(Intent.ACTION_PICK,
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(i, PICK_IMAGE);
        } else if (view.getId() == R.id.txtTakePhoto) {
            Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, CAMERA_REQUEST);
        }
    }

    public void goEditor() {
        Intent intent = new Intent(this, EditorActivity.class);
        this.startActivity(intent);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Globals.pickedImage = null;
        Globals.pickedImageGallery = null;
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            Globals.pickedImageRotated = 0;
            Globals.pickedImage = (Bitmap) data.getExtras().get("data");
            goEditor();
        } else if (requestCode == PICK_IMAGE && resultCode == Activity.RESULT_OK) {
            Uri selectedImageUri = data.getData();
            if (null != selectedImageUri) {
                String[] filePathColumn = {MediaStore.Images.Media.DATA};

                Cursor cursor = getContentResolver().query(selectedImageUri, filePathColumn, null, null, null);
                cursor.moveToFirst();

                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                String filePath = cursor.getString(columnIndex);
                cursor.close();

                Globals.pickedImageRotated = Utils.getCameraPhotoOrientation(MainActivity.this, selectedImageUri, filePath);
                Globals.pickedImageGallery = selectedImageUri;
                goEditor();
            }
        }
    }
}
