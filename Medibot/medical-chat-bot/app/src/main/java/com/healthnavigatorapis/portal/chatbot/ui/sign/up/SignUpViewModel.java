package com.healthnavigatorapis.portal.chatbot.ui.sign.up;

import android.app.Application;
import android.text.TextUtils;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseViewModel;

import androidx.annotation.NonNull;
import androidx.lifecycle.MutableLiveData;

public class SignUpViewModel extends BaseViewModel {
    private MutableLiveData<String> nameField = new MutableLiveData<>();
    private MutableLiveData<String> emailField = new MutableLiveData<>();
    private MutableLiveData<String> passwordField = new MutableLiveData<>();
    private MutableLiveData<String> confirmPasswordField = new MutableLiveData<>();

    public SignUpViewModel(@NonNull Application application) {
        super(application);
    }

    public void onSignUp() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    private boolean validateInputFields() {
        return !TextUtils.isEmpty(emailField.getValue()) && !TextUtils.isEmpty(passwordField.getValue());
    }

    public MutableLiveData<String> getNameField() {
        return nameField;
    }

    public MutableLiveData<String> getEmailField() {
        return emailField;
    }

    public MutableLiveData<String> getPasswordField() {
        return passwordField;
    }

    public MutableLiveData<String> getConfirmPasswordField() {
        return confirmPasswordField;
    }
}
