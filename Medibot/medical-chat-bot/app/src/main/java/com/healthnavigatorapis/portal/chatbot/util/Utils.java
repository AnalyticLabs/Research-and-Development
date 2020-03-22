package com.healthnavigatorapis.portal.chatbot.util;

import android.text.TextUtils;

public class Utils {
    public static int ageParse(String value) {
        if (!TextUtils.isEmpty(value)) {
            String age = value.replaceAll("\\D+", "");
            if (!TextUtils.isEmpty(age) && TextUtils.isDigitsOnly(age)) {
                return Integer.parseInt(age);
            } else {
                return -1;
            }
        }
        return -1;
    }
}
