package com.example.bymyself;


import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Build;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.RadioButton;
import android.widget.Switch;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_32S;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.resize;

/**
 * @author hirah
 */
public class CameraView extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private JavaCameraView mCameraView;
    private static int requestPermissionId = 1;
    private static int buildVersion = 23;
    private static int cameraId = 0;
    private static int process = 0;
    private static int colorToGray = 0;
    Mat mFrame;
    Mat mGray;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        // 动态权限
        if (Build.VERSION.SDK_INT >= buildVersion) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    requestPermissionId);
        }

        // 设置窗口
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // 初始化相机
        mCameraView = findViewById(R.id.cv_camera);
        mCameraView.setVisibility(SurfaceView.VISIBLE);
        mCameraView.setCvCameraViewListener(this);

        // 开始预览
        mCameraView.setCameraIndex(0);
        mCameraView.enableView();
        mCameraView.enableFpsMeter();
        // 切换摄像头方法
        RadioButton backOption = findViewById(R.id.backCameraOption);
        backOption.setChecked(true);
        backOption.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraId = (cameraId + 1) % 2;
                cameraSwitch(cameraId);
            }
        });

        Switch btHistogram = findViewById(R.id.sw_color1);
        btHistogram.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                process = (process + 1) % 2;
            }
        });

//        Switch btGray = findViewById(R.id.sw_color1);
//        btGray.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                colorToGray = (colorToGray + 1) % 2;
//            }
//        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mFrame = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mFrame.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mFrame = inputFrame.rgba();
        mFrame = process(mFrame);
        return mFrame;
    }

    /*
     ** Frame process method
     */

    private Mat process(Mat frame) {
//        if (colorToGray == 1) {
//            new NDKInterface().colorToGray(frame.getNativeObjAddr(), mGray.getNativeObjAddr());
//            return mGray;
//        }
          Bitmap image1 = BitmapFactory.decodeResource(this.getResources(), R.drawable.moss);
          Mat mat1 = new Mat(image1.getHeight()/4,image1.getWidth()/4,CvType.CV_8UC4);
          Utils.bitmapToMat(image1,mat1);
          resize(mat1, mat1, new Size(200,200));

//        if (process == 1) {
//            new NDKInterface().histogram(frame.getNativeObjAddr(),mGray.getNativeObjAddr());
//            return mGray;
//       }
//          if(process==1)      //ORB匹配
//          {
//              new NDKInterface().OEBMatch(frame.getNativeObjAddr(),mat1.getNativeObjAddr(),mGray.getNativeObjAddr());
//              return mGray;
//          }
//        if(process==1)         //LK光流法物体跟踪
//        {
//            new NDKInterface().opticalflow(frame.getNativeObjAddr(),mGray.getNativeObjAddr());
//            return mGray;
//        }


//        if(process==1)
//        {
//            new NDKInterface().trace(frame.getNativeObjAddr(),mat1.getNativeObjAddr(),mGray.getNativeObjAddr());
//            return mGray;
//        }
        if(process==1)
        {
            new NDKInterface().trace11(frame.getNativeObjAddr(),mat1.getNativeObjAddr(),mGray.getNativeObjAddr());
            return mGray;
        }
//        if(process==2)
//        {
//            new NDKInterface().mtf(frame.getNativeObjAddr(),mat1.getNativeObjAddr(),mGray.getNativeObjAddr());
//            return mGray;
//        }
//        if(process == 1)
//        {
//            new NDKInterface().tem(frame.getNativeObjAddr(),mat1.getNativeObjAddr());
//            return frame;
//        }

        return frame;
    }

    private void cameraSwitch(int id) {
        cameraId = id;
        mCameraView.setCameraIndex(cameraId);
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        mCameraView.enableView();
        mCameraView.enableFpsMeter();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mCameraView != null) {
            mCameraView.setCameraIndex(cameraId);
            mCameraView.enableView();
            mCameraView.enableFpsMeter();
        }
    }
}
