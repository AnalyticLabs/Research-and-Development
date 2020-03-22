package com.healthnavigatorapis.portal.chatbot.ui.login;

import android.os.Bundle;
import android.view.View;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragmentViewModel;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentLoginBinding;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.ViewModelProviders;

public class LoginFragment extends BaseFragmentViewModel<FragmentLoginBinding, LoginViewModel> {

    public static LoginFragment newInstance() {
        LoginFragment fragment = new LoginFragment();

        fragment.setArguments(new Bundle());
        return fragment;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_login;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public LoginViewModel initViewModel() {
        return ViewModelProviders.of(this).get(LoginViewModel.class);
    }

    @Override
    public int getBindingVariable() {
        return BR.viewModel;
    }

    @Override
    protected void setSubscribers() {
        getViewModel().getNavigation().observe(this, this::navigationController);
    }

    private void navigationController(FragmentNavigator.Navigation navigation) {
        switch (navigation) {
            case SignIn:
                if (getNavigator() != null) {
                    getNavigator().showSignInFragment();
                }
                break;
            case SignUp:
                if (getNavigator() != null) {
                    getNavigator().showSignUpFragment();
                }
                break;
            default:
                break;
        }
    }

}
