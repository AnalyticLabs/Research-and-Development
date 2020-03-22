package com.healthnavigatorapis.portal.chatbot.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;
import com.healthnavigatorapis.portal.chatbot.databinding.ItemCauseBinding;

import net.cachapa.expandablelayout.ExpandableLayout;

import androidx.databinding.DataBindingUtil;

public class CauseLayout extends ExpandableLayout {

    private ItemCauseBinding mBinding;
    private Cause mData;

    public CauseLayout(Context context) {
        super(context);
        init();
    }

    public CauseLayout(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }


    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
    }

    private void init() {
        setDuration(Constants.DELAY_MILLI);
        mBinding = DataBindingUtil.inflate(LayoutInflater.from(getContext()), R.layout.item_cause,
                this, false);
        addView(mBinding.getRoot());
    }

    public Cause getData() {
        return mData;
    }

    public void setData(Cause data, int count) {
        mData = data;
        mBinding.setCause(data);
        mBinding.setCount(count);
        mBinding.executePendingBindings();
    }

}
