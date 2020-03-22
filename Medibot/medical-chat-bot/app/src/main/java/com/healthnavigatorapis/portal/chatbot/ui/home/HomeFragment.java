package com.healthnavigatorapis.portal.chatbot.ui.home;

import android.os.Bundle;
import android.view.View;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragmentViewModel;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentHomeBinding;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.ViewModelProviders;

public class HomeFragment extends BaseFragmentViewModel<FragmentHomeBinding, HomeViewModel> {

    public static HomeFragment newInstance() {
        HomeFragment fragment = new HomeFragment();
        fragment.setArguments(new Bundle());
        return fragment;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_home;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        getBinding().setLifecycleOwner(this);
    }

    @Override
    public HomeViewModel initViewModel() {
        return ViewModelProviders.of(this).get(HomeViewModel.class);
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
            case ChatBot:
                if (getNavigator() != null) {
                    getNavigator().showChatBotFragment();
                }
        }
    }


}
