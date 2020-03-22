package com.healthnavigatorapis.portal.chatbot.ui.main;

public interface FragmentNavigator {
    void showSplashFragment();

    void showLoginFragment();

    void showSignInFragment();

    void showSignUpFragment();

    void showChatBotFragment();

    void showHomeFragment();

    enum Navigation {
        Splash,
        Login,
        SignIn,
        SignUp,
        ChatBot,
        Home,
        Back
    }
}
