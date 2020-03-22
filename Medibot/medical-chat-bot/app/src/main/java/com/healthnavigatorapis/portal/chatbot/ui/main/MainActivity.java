package com.healthnavigatorapis.portal.chatbot.ui.main;

import android.os.Bundle;

import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseActivity;
import com.healthnavigatorapis.portal.chatbot.databinding.ActivityMainBinding;
import com.healthnavigatorapis.portal.chatbot.ui.chat.ChatFragment;
import com.healthnavigatorapis.portal.chatbot.ui.home.HomeFragment;
import com.healthnavigatorapis.portal.chatbot.ui.login.LoginFragment;
import com.healthnavigatorapis.portal.chatbot.ui.sign.in.SignInFragment;
import com.healthnavigatorapis.portal.chatbot.ui.sign.up.SignUpFragment;
import com.healthnavigatorapis.portal.chatbot.ui.splash.SplashFragment;

public class MainActivity extends BaseActivity<ActivityMainBinding> implements FragmentNavigator {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (savedInstanceState == null) {
            showSplashFragment();
        }
    }

    @Override
    protected int getLayoutId() {
        return R.layout.activity_main;
    }

    @Override
    public void showSplashFragment() {
        getSupportFragmentManager()
                .beginTransaction()
                .add(R.id.mainContainer, SplashFragment.newInstance())
                .commit();
    }

    @Override
    public void showLoginFragment() {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.mainContainer, LoginFragment.newInstance())
                .commit();
    }

    @Override
    public void showSignInFragment() {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.mainContainer, SignInFragment.newInstance())
                .addToBackStack(null)
                .commit();
    }

    @Override
    public void showSignUpFragment() {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.mainContainer, SignUpFragment.newInstance())
                .addToBackStack(null)
                .commit();
    }

    @Override
    public void showChatBotFragment() {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.mainContainer, ChatFragment.newInstance())
                .addToBackStack(null)
                .commit();
    }

    @Override
    public void showHomeFragment() {
        getSupportFragmentManager().popBackStack();
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.mainContainer, HomeFragment.newInstance())
                .commit();
    }
}
