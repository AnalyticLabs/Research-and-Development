package com.healthnavigatorapis.portal.chatbot.util;

import android.util.Log;

import com.healthnavigatorapis.portal.chatbot.BuildConfig;


public class Logger {

    private static String getClassName(Object object) {
        String mClassName = object.getClass().getName();
        int mFirstPosition = mClassName.lastIndexOf(".") + 1;
        if (mFirstPosition < 0) {
            mFirstPosition = 0;
        }
        mClassName = mClassName.substring(mFirstPosition);
        mFirstPosition = mClassName.lastIndexOf("$");
        if (mFirstPosition > 0) {
            mClassName = mClassName.substring(0, mFirstPosition);
        }
        return mClassName;
    }

    public static void e(Object object, String message) {
        if (BuildConfig.DEBUG)
            Log.e((object instanceof String) ? (String) object : getClassName(object), message);
    }

    public static void e(Object object, String message, Exception exception) {
        if (BuildConfig.DEBUG)
            Log.e((object instanceof String) ? (String) object : getClassName(object), message, exception);
    }

    public static void i(Object object, String message) {
        if (BuildConfig.DEBUG)
            Log.i((object instanceof String) ? (String) object : getClassName(object), message);
    }

    public static void v(Object object, String message) {
        if (BuildConfig.DEBUG)
            Log.v((object instanceof String) ? (String) object : getClassName(object), message);
    }

    public static void w(Object object, String message) {
        if (BuildConfig.DEBUG)
            Log.w((object instanceof String) ? (String) object : getClassName(object), message);
    }

    public static void w(Object object, String message, Exception exception) {
        if (BuildConfig.DEBUG)
            Log.w((object instanceof String) ? (String) object : getClassName(object), message, exception);
    }
}