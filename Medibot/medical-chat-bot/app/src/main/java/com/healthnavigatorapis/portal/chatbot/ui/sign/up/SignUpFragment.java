package com.healthnavigatorapis.portal.chatbot.ui.sign.up;

import android.os.Bundle;
import android.view.View;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragmentViewModel;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentSignUpBinding;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.ViewModelProviders;

public class SignUpFragment extends BaseFragmentViewModel<FragmentSignUpBinding, SignUpViewModel> {

    public static SignUpFragment newInstance() {
        SignUpFragment fragment = new SignUpFragment();

        fragment.setArguments(new Bundle());
        return fragment;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_sign_up;
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
    public SignUpViewModel initViewModel() {
        return ViewModelProviders.of(this).get(SignUpViewModel.class);
    }

    @Override
    public int getBindingVariable() {
        return BR.viewModel;
    }

    @Override
    protected void setSubscribers() {

    }
}
