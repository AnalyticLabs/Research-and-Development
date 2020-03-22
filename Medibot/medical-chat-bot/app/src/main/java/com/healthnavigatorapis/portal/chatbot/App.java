package com.healthnavigatorapis.portal.chatbot;

import android.app.Application;

import com.crashlytics.android.Crashlytics;
import com.healthnavigatorapis.portal.chatbot.data.DataRepository;
import com.healthnavigatorapis.portal.chatbot.data.local.AppDatabase;
import com.healthnavigatorapis.portal.chatbot.data.remote.ApiClient;
import com.healthnavigatorapis.portal.chatbot.data.remote.service.HealthService;

import io.fabric.sdk.android.Fabric;

public class App extends Application {

    public AppDatabase getDatabase() {
        return AppDatabase.getInstance(this);
    }

    public HealthService getService() {
        return ApiClient.getClient().create(HealthService.class);
    }

    public DataRepository getRepository() {
        return DataRepository.getInstance(getDatabase(), getService());
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Fabric.with(this, new Crashlytics());
    }
}
