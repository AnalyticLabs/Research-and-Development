package com.healthnavigatorapis.portal.chatbot.data.local.entity;

import android.text.TextUtils;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.data.local.model.IChatUser;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.IUserData;

import java.util.Calendar;
import java.util.Date;

import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity(tableName = "user_table")
public class User implements IChatUser, IUserData {
    @PrimaryKey
    private int id;
    private String name;
    private int age;
    private String gender;

    public User() {
    }

    @Ignore
    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String getChatName() {
        if (!TextUtils.isEmpty(name)) {
            return String.valueOf(name.charAt(0)).toUpperCase();
        } else {
            return String.valueOf("U");
        }
    }

    @Override
    public UserType getType() {
        if (id == Constants.BOT_ID) {
            return UserType.BOT;
        } else if (id == Constants.USER_ID) {
            return UserType.MAIN_USER;
        } else {
            return UserType.SOMEONE_ELSE;
        }
    }

    @Override
    public int geAgeInDays() {
        Calendar cal = Calendar.getInstance();
        Date today = cal.getTime();
        cal.set(cal.get(Calendar.YEAR) - age, Calendar.JANUARY, 1);
        Date birthday = cal.getTime();

        long dateSubtract = today.getTime() - birthday.getTime();
        long time = 1000 * 60 * 60 * 24;
        return (int) (dateSubtract / time);
    }

    @Override
    public String getGender() {
        if (!TextUtils.isEmpty(gender)) {
            return String.valueOf(gender.charAt(0)).toUpperCase();
        } else {
            return String.valueOf("B");
        }
    }

    public String getFullGender() {
        if (getGender().equalsIgnoreCase("m")) {
            return "Male";
        } else if (getGender().equalsIgnoreCase("female")) {
            return "Female";
        }
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    @Override
    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    @Override
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }
}
