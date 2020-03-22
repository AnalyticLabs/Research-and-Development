package com.healthnavigatorapis.portal.chatbot.data.local;

import android.content.Context;

import com.healthnavigatorapis.portal.chatbot.data.local.dao.UserDao;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

@Database(entities = {User.class}, version = AppDatabase.VERSION, exportSchema = false)
public abstract class AppDatabase extends RoomDatabase {
    static final int VERSION = 1;
    private static AppDatabase mInstance;

    public static AppDatabase getInstance(Context context) {
        if (mInstance == null) {
            synchronized (AppDatabase.class) {
                if (mInstance == null) {
                    mInstance = Room.databaseBuilder(context.getApplicationContext(),
                            AppDatabase.class, "bot_chat_database.db")
                            .build();
                }
            }
        }
        return mInstance;
    }

    public abstract UserDao getUserDao();
}
