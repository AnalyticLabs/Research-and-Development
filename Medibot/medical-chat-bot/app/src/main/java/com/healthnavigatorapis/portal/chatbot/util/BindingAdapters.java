package com.healthnavigatorapis.portal.chatbot.util;

import android.text.Html;
import android.view.View;
import android.widget.TextView;

import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;
import com.healthnavigatorapis.portal.chatbot.ui.view.CauseListLayout;
import com.healthnavigatorapis.portal.chatbot.ui.view.ChoiceListLayout;

import java.util.List;

import androidx.databinding.BindingAdapter;

public class BindingAdapters {

    @BindingAdapter({"app:setChoice"})
    public static void setChoices(ChoiceListLayout choiceListLayout, Choices choices) {
        if (choices != null) {
            choiceListLayout.setVisibility(View.VISIBLE);
            choiceListLayout.setData(choices);
        } else {
            choiceListLayout.setVisibility(View.GONE);
        }
    }

    @BindingAdapter({"app:setCause"})
    public static void setCause(CauseListLayout causeListLayout, List<Cause> causes) {
        if (causes != null && !causes.isEmpty()) {
            causeListLayout.setVisibility(View.VISIBLE);
            causeListLayout.setData(causes);
        }
    }

    @BindingAdapter({"app:textHtml"})
    public static void setText(TextView textView, String text) {
        textView.setText(Html.fromHtml(text));
    }

    @BindingAdapter({"app:setColorScheme"})
    public static void setColorScheme(View view, int rawScore) {
        if (rawScore < 20) {
            view.setBackgroundColor(view.getContext().getResources().getColor(android.R.color.darker_gray));
        } else if (rawScore < 35) {
            view.setBackgroundColor(view.getContext().getResources().getColor(android.R.color.holo_green_light));
        } else if (rawScore < 75) {
            view.setBackgroundColor(view.getContext().getResources().getColor(android.R.color.holo_orange_light));
        } else if (rawScore < 95) {
            view.setBackgroundColor(view.getContext().getResources().getColor(android.R.color.holo_red_light));
        } else {
            view.setBackgroundColor(view.getContext().getResources().getColor(android.R.color.holo_red_dark));
        }
    }
}
