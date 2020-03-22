package com.healthnavigatorapis.portal.chatbot.ui.chat;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.speech.RecognizerIntent;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.inputmethod.InputMethodManager;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseFragmentViewModel;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Message;
import com.healthnavigatorapis.portal.chatbot.databinding.FragmentChatBinding;
import com.healthnavigatorapis.portal.chatbot.interfaces.IChoicesPressed;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.ViewModelProviders;
import androidx.recyclerview.widget.LinearLayoutManager;

import static android.app.Activity.RESULT_OK;

public class ChatFragment extends BaseFragmentViewModel<FragmentChatBinding, ChatViewModel> implements IChoicesPressed {

    private ChatAdapter mAdapter;
    private LinearLayoutManager mManager;

    public static ChatFragment newInstance() {
        ChatFragment fragment = new ChatFragment();
        fragment.setArguments(new Bundle());
        return fragment;
    }

    public static void hideKeyboardFrom(Context context, View view) {
        InputMethodManager imm = (InputMethodManager) context.getSystemService(Activity.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }

    @Override
    protected int getLayoutId() {
        return R.layout.fragment_chat;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        initAdapter();
    }

    private void initAdapter() {
        mAdapter = new ChatAdapter();
        mAdapter.setListener(this);

        mManager = new LinearLayoutManager(getContext());
        mManager.setStackFromEnd(true);

        getBinding().chatContent.setLayoutManager(mManager);
        getBinding().chatContent.setAdapter(mAdapter);
        getBinding().setLifecycleOwner(this);

        getBinding().chatContent.setItemViewCacheSize(20);
        getBinding().chatContent.getRecycledViewPool().setMaxRecycledViews(0, 20);
        getBinding().chatContent.getRecycledViewPool().setMaxRecycledViews(1, 20);
    }

    @Override
    public ChatViewModel initViewModel() {
        return ViewModelProviders.of(this).get(ChatViewModel.class);
    }

    @Override
    public int getBindingVariable() {
        return BR.viewModel;
    }

    private void addMessage(Message message) {
        mAdapter.addMessage(message);
        onScroll();
    }

    @Override
    protected void setSubscribers() {
        getViewModel().getMessageLiveData().observe(this, this::addMessage);
        getViewModel().getVoiceRecognition().observe(this, aVoid -> getSpeechInput());
        getViewModel().getRefreshPressed().observe(this, aVoid -> onRefresh());
        getViewModel().getIsChoices().observe(this, this::isChoices);
        getViewModel().getMessageList().observe(this, messages -> {
            for (Message message : messages) {
                addMessage(message);
            }

        });
        getViewModel().getError().observe(this, this::showError);
        getViewModel().getNavigation().observe(this, this::navigationController);
        getViewModel().getIsShowInput().observe(this, aBoolean -> hideKeyboardFrom(getContext(), getView()));
    }

    private void onScroll() {
        getBinding().chatContent.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            public void onGlobalLayout() {
                getBinding().chatContent.smoothScrollToPosition(mAdapter.getItemCount() - 1);
                getBinding().chatContent.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            }
        });
    }

    private void showError(String error) {
        Toast.makeText(getContext(), error, Toast.LENGTH_SHORT).show();
    }

    private void onRefresh() {
        mAdapter.clearMessages();
    }

    private void isChoices(boolean enable) {
        hideKeyboardFrom(getContext(), getView());
        getBinding().chatInputText.setVisibility(enable ? View.GONE : View.VISIBLE);
        getBinding().chatSendText.setVisibility(enable ? View.GONE : View.VISIBLE);
        getBinding().chatInputVoice.setEnabled(!enable);
        getBinding().chatArrow.setVisibility(enable ? View.INVISIBLE : View.VISIBLE);
    }

    @Override
    public void onChoicePressed(List<Choice> selectedChoices) {
        getViewModel().selectedChoice(selectedChoices);
    }

    public void getSpeechInput() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_WEB_SEARCH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US");
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "en-US");

        if (intent.resolveActivity(getActivity().getPackageManager()) != null) {
            startActivityForResult(intent, Constants.RESULT_SPEECH_RECOGNITION);
        } else {
            Toast.makeText(getContext(), "Your device isn't supported speech input", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case Constants.RESULT_SPEECH_RECOGNITION:
                if (resultCode == RESULT_OK && data != null) {
                    ArrayList<String> result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                    getViewModel().onSendMessage(result.get(0), true);
                }
                break;
        }
    }

    public void onBackPressed() {
        getFragmentManager().popBackStack();
    }

    private void navigationController(FragmentNavigator.Navigation navigation) {
        switch (navigation) {
            case Back:
                onBackPressed();
                break;
        }
    }

}
