package com.healthnavigatorapis.portal.chatbot.ui.splash;

import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;

import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragment;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentSplashBinding;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class SplashFragment extends BaseFragment<FragmentSplashBinding> {

    public static SplashFragment newInstance() {
        SplashFragment fragment = new SplashFragment();

        fragment.setArguments(new Bundle());
        return fragment;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_splash;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        String[] languages = {"Choose Language", "English"};
        ArrayAdapter<CharSequence> adapter = new ArrayAdapter<>(getContext(), R.layout.item_language, R.id.language, languages);
        getBinding().splashSpinner.setAdapter(adapter);
    }

    @Override
    public void onResume() {
        super.onResume();
        getBinding().setPresenter(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        getBinding().setPresenter(null);
    }

    public void onNext() {
        if (getNavigator() != null) {
            getNavigator().showLoginFragment();
        }
    }
}
