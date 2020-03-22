package com.healthnavigatorapis.portal.chatbot.ui.sign.in;

import android.app.Application;
import android.text.TextUtils;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.App;
import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseViewModel;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.lifecycle.MutableLiveData;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;

public class SignInViewModel extends BaseViewModel {
    private MutableLiveData<String> loginField = new MutableLiveData<>();
    private MutableLiveData<String> passwordField = new MutableLiveData<>();
    private Disposable disposable;

    public SignInViewModel(@NonNull Application application) {
        super(application);
    }

    public void onSubmit() {
        if (validateInputFields()) {
            User user = new User(Constants.USER_ID, "Sibasis");
            user.setAge(24);
            user.setGender("Male");
            disposable = ((App) getApplication()).getRepository()
                    .insertUser(user)
                    .subscribeOn(Schedulers.io())
                    .observeOn(AndroidSchedulers.mainThread())
                    .subscribe(id -> setNavigation(FragmentNavigator.Navigation.Home));
        } else {
            Toast.makeText(getApplication(), "Please input the login and password", Toast.LENGTH_SHORT).show();
        }
    }

    public void onForgotPassword() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    private boolean validateInputFields() {
        return !TextUtils.isEmpty(loginField.getValue()) && !TextUtils.isEmpty(passwordField.getValue());
    }

    public MutableLiveData<String> getLoginField() {
        return loginField;
    }

    public MutableLiveData<String> getPasswordField() {
        return passwordField;
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        if (disposable != null) {
            disposable.dispose();
        }
    }
}
