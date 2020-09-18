package com.example.bymyself;

import android.graphics.Bitmap;

public class NDKInterface {
    static{
        System.loadLibrary("native-lib");
    }
    public native void getEdge(Object bitmap);
    public native int colorToGray(long srcAdd, long dstAdd);
    public native int histogram(long srcAdd,long dstAdd);
    public native int OEBMatch(long srcAdd,long temAdd,long dstAdd);
    public native void tem(long srcAdd,long dstAdd);
    public native void opticalflow(long srcAdd,long dstAdd);
    public native void trace(long srcAdd,long temAdd,long dstAdd);

    public native void trace11(long nativeObjAddr, long nativeObjAddr1, long nativeObjAddr2);
    public native void mtf(long nativeObjAddr, long nativeObjAddr1, long nativeObjAddr2);

}
