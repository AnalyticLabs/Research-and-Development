package com.healthnavigatorapis.portal.chatbot.data.local.dao;

import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import io.reactivex.Single;

@Dao
public interface UserDao {

    @Query("SELECT * FROM user_table WHERE id = :userId")
    Single<User> getUser(int userId);

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    Long insert(User user);
}
