package com.healthnavigatorapis.portal.chatbot.ui.home;

import android.app.Application;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.App;
import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseViewModel;
import com.healthnavigatorapis.portal.chatbot.data.DataRepository;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.LiveDataReactiveStreams;

public class HomeViewModel extends BaseViewModel {
    private DataRepository mRepository;
    private LiveData<User> user;

    public HomeViewModel(@NonNull Application application) {
        super(application);
        mRepository = ((App) getApplication()).getRepository();
        user = LiveDataReactiveStreams.fromPublisher(mRepository.getUser(Constants.USER_ID).toFlowable());
    }

    public LiveData<User> getUser() {
        return user;
    }

    public void onSettings() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onAskQuestions() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onUtilities() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onAboutSettings() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onSymptomsCheck() {
        setNavigation(FragmentNavigator.Navigation.ChatBot);
    }

    public void onProfile() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onCleared() {
        super.onCleared();
    }
}
