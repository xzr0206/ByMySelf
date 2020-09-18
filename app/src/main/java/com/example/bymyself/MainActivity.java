package com.example.bymyself;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Dialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity {
    private static int cannyID = 0;
    private static final String TAG = "OpenCV";
    static {
        System.loadLibrary("native-lib");
    }
    // OpenCV库静态加载并初始化
    static {
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "Load successfully...");
        } else {
            Log.i(TAG, "Fail to load...");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btCanny = findViewById(R.id.bt_toCanny);
        btCanny.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cannyID = (cannyID + 1) % 2;
                imageToCanny();
            }
        });

        Button btCamera = findViewById(R.id.bt_camera);
        btCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this.getApplicationContext(), CameraView.class);
                showDialog(0);
                startActivity(intent);
            }
        });

    }
    @Override
    protected Dialog onCreateDialog(int id) {
        Dialog dialog = new Dialog(this);
        return dialog;
        //  return super.onCreateDialog(id);
    }




//    private void imageToCanny() {
//        ImageView imageView = findViewById(R.id.imageView);
//        Bitmap image = BitmapFactory.decodeResource(this.getResources(), R.drawable.luffy);
//        if (cannyID == 1) {
//            new NDKInterface().getEdge(image);
//        }
//        imageView.setImageBitmap(image);
//    }

    private void imageToCanny() {
        ImageView imageView = findViewById(R.id.imageView);
        Bitmap image1 = BitmapFactory.decodeResource(this.getResources(), R.drawable.luffy);
        Bitmap image2= BitmapFactory.decodeResource(this.getResources(), R.drawable.luffyss);
        Mat mat1 = new Mat(image1.getHeight(),image1.getWidth(),CvType.CV_8UC4);
        Mat mat2 = new Mat(image2.getHeight(),image2.getWidth(),CvType.CV_8UC4);
        Utils.bitmapToMat(image1,mat1);
        Utils.bitmapToMat(image2,mat2);
        System.out.println(mat1);
        System.out.println(mat2);
        if (cannyID == 1) {
            new NDKInterface().tem(mat1.getNativeObjAddr(),mat2.getNativeObjAddr());
        }
        Utils.matToBitmap(mat1,image1);
        imageView.setImageBitmap(image1);
    }



}
