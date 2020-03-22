package com.healthnavigatorapis.portal.chatbot.ui.view;

import android.content.Context;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.widget.LinearLayout;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;

import net.cachapa.expandablelayout.ExpandableLayout;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatButton;

public class CauseListLayout extends LinearLayout {

    private List<Cause> mCauses = new ArrayList<>();
    private AppCompatButton mButton;
    private boolean isMore = true;

    public CauseListLayout(Context context) {
        super(context);
        init();
    }

    public CauseListLayout(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setOrientation(LinearLayout.VERTICAL);
    }

    public void setData(List<Cause> causes) {
        if (mCauses.isEmpty()) {
            ((ExpandableLayout) getParent()).setDuration(Constants.DELAY_MILLI);
            ((ExpandableLayout) getParent()).expand(true);
            mCauses = causes;
            removeAllViews();

            for (int i = 0; i < causes.size(); i++) {
                if (i < 5) {
                    addCause(causes.get(i), i + 1, true);
                } else {
                    addCause(causes.get(i), i + 1, false);
                }
            }

            mButton = new AppCompatButton(getContext());
            mButton.setText(getResources().getString(R.string.bot_more_button));
            mButton.setTextColor(getResources().getColor(android.R.color.background_dark));
            mButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
            LayoutParams layoutParams = new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT);
            layoutParams.setMargins(dpToPx(16), 0, dpToPx(16), dpToPx(8));
            mButton.setLayoutParams(layoutParams);
            mButton.setBackground(getResources().getDrawable(R.drawable.button_selector));
            mButton.setClickable(true);
            mButton.setFocusable(true);
            mButton.setOnClickListener((view) -> onShowItems());
            addView(mButton);
        }
    }

    public void addCause(Cause data, int count, boolean isShow) {
        CauseLayout causeLayout = new CauseLayout(getContext());
        causeLayout.setData(data, count);
        if (isShow) {
            causeLayout.setVisibility(VISIBLE);
            causeLayout.setExpanded(true, false);
        } else {
            causeLayout.setVisibility(GONE);
        }
        addView(causeLayout);
    }

    public void onShowItems() {
        for (int i = 5; i < mCauses.size(); i++) {
            CauseLayout causeLayout = (CauseLayout) getChildAt(i);
            causeLayout.setVisibility(isMore ? VISIBLE : GONE);
            causeLayout.setExpanded(isMore, false);
        }
        isMore = !isMore;
        mButton.setText(isMore ? getResources().getString(R.string.bot_more_button) : getResources().getString(R.string.bot_hide_more_button));
    }

    private int dpToPx(float dp) {
        return (int) TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, dp,
                getResources().getDisplayMetrics());
    }
}