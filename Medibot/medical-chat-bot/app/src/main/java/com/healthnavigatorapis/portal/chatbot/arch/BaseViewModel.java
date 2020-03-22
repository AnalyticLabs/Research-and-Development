package com.healthnavigatorapis.portal.chatbot.arch;

import android.app.Application;

import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;

public abstract class BaseViewModel extends AndroidViewModel {
    private final SingleLiveEvent<FragmentNavigator.Navigation> mNavigation = new SingleLiveEvent<>();

    public BaseViewModel(@NonNull Application application) {
        super(application);
    }


    public SingleLiveEvent<FragmentNavigator.Navigation> getNavigation() {
        return mNavigation;
    }

    public void setNavigation(FragmentNavigator.Navigation navigation) {
        mNavigation.setValue(navigation);
    }

}
