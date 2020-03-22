package com.healthnavigatorapis.portal.chatbot.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.LinearLayout;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;
import com.healthnavigatorapis.portal.chatbot.interfaces.IChoiceLayoutPressed;
import com.healthnavigatorapis.portal.chatbot.interfaces.IChoicesPressed;

import net.cachapa.expandablelayout.ExpandableLayout;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatButton;

public class ChoiceListLayout extends LinearLayout implements IChoiceLayoutPressed {

    private Choices mChoices;
    private IChoicesPressed mListener;
    private ArrayList<Choice> mChoiceList = new ArrayList<>();
    private AppCompatButton mButton;
    private boolean mChoicePressed;

    public ChoiceListLayout(Context context) {
        super(context);
        init();
    }

    public ChoiceListLayout(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setOrientation(LinearLayout.VERTICAL);
    }

    public void setData(Choices choices) {
        ((ExpandableLayout) getParent()).setDuration(Constants.DELAY_MILLI);
        mChoices = choices;
        removeAllViews();

        for (Choice choice : choices.getChoiceList()) {
            addChoice(choice);
        }
        if (mChoices.getType() == Choices.ChoiceType.MULTIPLE && !choices.getChoiceList().isEmpty()) {
            mButton = new AppCompatButton(getContext());
            mButton.setText(getResources().getString(R.string.bot_accept_none));
            mButton.setTextColor(getResources().getColor(android.R.color.background_dark));
            mButton.setBackground(getResources().getDrawable(R.drawable.button_selector));
            mButton.setClickable(true);
            mButton.setFocusable(true);
            mButton.setOnClickListener((view) -> onSendChoices());
            addView(mButton);
        }
        ((ExpandableLayout) getParent()).expand(true);
    }

    public void addChoice(Choice data) {
        ChoiceLayout choiceLayout = new ChoiceLayout(getContext());
        choiceLayout.setData(data, mChoices.getType());
        choiceLayout.setListener(this);

        addView(choiceLayout);
    }

    public void setListener(IChoicesPressed listener) {
        mListener = listener;
    }


    public void hideChoice() {
        ((ExpandableLayout) getParent()).collapse(true);
        mChoices.getChoiceList().clear();
        mChoicePressed = false;
    }

    @Override
    public void onChoicePressed(Choice choice, Choices.ChoiceType type) {
        if (mListener != null) {
            if (type == Choices.ChoiceType.SINGLE && !mChoicePressed) {
                mChoicePressed = true;
                ArrayList<Choice> choices = new ArrayList<>();
                choices.add(choice);
                mListener.onChoicePressed(choices);
                hideChoice();
            } else {
                if (choice.isSelected()) {
                    mChoiceList.add(choice);
                } else {
                    mChoiceList.remove(choice);
                }
                if (mChoiceList.isEmpty()) {
                    mButton.setText(getResources().getString(R.string.bot_accept_none));
                } else {
                    mButton.setText(getResources().getString(R.string.bot_accept_button));
                }
            }
        }
    }

    public void onSendChoices() {
        mButton.setEnabled(false);
        List<Choice> choices = new ArrayList<>(mChoiceList);
        mChoiceList.clear();
        if (mListener != null) {
            mListener.onChoicePressed(choices);
        }
        hideChoice();
    }
}