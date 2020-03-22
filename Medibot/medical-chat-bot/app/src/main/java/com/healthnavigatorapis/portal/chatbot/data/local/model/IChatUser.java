package com.healthnavigatorapis.portal.chatbot.data.local.model;

public interface IChatUser {
    int getId();

    String getName();

    String getChatName();

    UserType getType();

    int getAge();

    String getGender();

    enum UserType {
        MAIN_USER,
        SOMEONE_ELSE,
        BOT
    }
}
