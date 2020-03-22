package com.healthnavigatorapis.portal.chatbot.data.remote.model;

public interface IUserData {
    String getName();

    int geAgeInDays();

    String getGender();

    void setGender(String gender);

    void setAge(int age);
}
