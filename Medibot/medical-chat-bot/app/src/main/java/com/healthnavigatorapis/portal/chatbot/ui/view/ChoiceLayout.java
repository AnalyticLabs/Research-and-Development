package com.healthnavigatorapis.portal.chatbot.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.RelativeLayout;

import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;
import com.healthnavigatorapis.portal.chatbot.databinding.ItemChoiceBinding;
import com.healthnavigatorapis.portal.chatbot.interfaces.IChoiceLayoutPressed;

import androidx.databinding.DataBindingUtil;

public class ChoiceLayout extends RelativeLayout {

    private ItemChoiceBinding mBinding;
    private Choice mData;
    private IChoiceLayoutPressed mListener;
    private Choices.ChoiceType mType;

    public ChoiceLayout(Context context) {
        super(context);
        init();
    }

    public ChoiceLayout(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public ChoiceLayout(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        mBinding.setPresenter(this);
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        mBinding.setPresenter(null);
    }

    private void init() {
        mBinding = DataBindingUtil.inflate(LayoutInflater.from(getContext()), R.layout.item_choice,
                this, false);
        addView(mBinding.getRoot());
    }

    public Choice getData() {
        return mData;
    }

    public void setData(Choice data, Choices.ChoiceType type) {
        mData = data;
        mType = type;
        mBinding.setData(data);
        mBinding.setIsMultiChoices(type == Choices.ChoiceType.MULTIPLE);
        mBinding.executePendingBindings();
        if (type == Choices.ChoiceType.SINGLE) {
            mBinding.choiceLayout.setOnClickListener((view) -> onChoiceClick(true));
        } else {
            mBinding.choiceLayout.setClickable(false);
            mBinding.choiceLayout.setFocusable(false);
            mBinding.choiceCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> onChoiceClick(isChecked));
        }
    }

    public void setListener(IChoiceLayoutPressed listener) {
        mListener = listener;
    }

    public void onChoiceClick(boolean isChecked) {
        if (mListener != null) {
            mData.setSelected(isChecked);
            mListener.onChoicePressed(mData, mType);
        }
    }
}
