package com.healthnavigatorapis.portal.chatbot.interfaces;

import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;

public interface IChoiceLayoutPressed {
    void onChoicePressed(Choice choice, Choices.ChoiceType type);
}
