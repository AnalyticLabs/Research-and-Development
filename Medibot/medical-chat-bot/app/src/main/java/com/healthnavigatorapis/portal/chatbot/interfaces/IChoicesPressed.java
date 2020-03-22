package com.healthnavigatorapis.portal.chatbot.interfaces;

import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;

import java.util.List;

public interface IChoicesPressed {
    void onChoicePressed(List<Choice> selectedChoices);
}
