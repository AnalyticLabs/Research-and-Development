package com.healthnavigatorapis.portal.chatbot.ui.login;

import android.app.Application;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseViewModel;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;

public class LoginViewModel extends BaseViewModel {

    public LoginViewModel(@NonNull Application application) {
        super(application);
    }

    public void onSingIn() {
        setNavigation(FragmentNavigator.Navigation.SignIn);
    }

    public void onSignUp() {
        setNavigation(FragmentNavigator.Navigation.SignUp);
    }

    public void onSignFacebook() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onSignGoogle() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }
}
