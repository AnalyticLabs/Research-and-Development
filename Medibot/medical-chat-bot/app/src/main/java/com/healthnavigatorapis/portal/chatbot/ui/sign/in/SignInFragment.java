package com.healthnavigatorapis.portal.chatbot.ui.sign.in;

import android.os.Bundle;
import android.view.View;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragmentViewModel;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentSignInBinding;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.ViewModelProviders;

public class SignInFragment extends BaseFragmentViewModel<FragmentSignInBinding, SignInViewModel> {

    public static SignInFragment newInstance() {
        SignInFragment fragment = new SignInFragment();

        fragment.setArguments(new Bundle());
        return fragment;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_sign_in;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        getBinding().setLifecycleOwner(this);
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public SignInViewModel initViewModel() {
        return ViewModelProviders.of(this).get(SignInViewModel.class);
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
            case Home:
                if (getNavigator() != null) {
                    getNavigator().showHomeFragment();
                }
                break;
            default:
                break;
        }
    }
}
