package com.healthnavigatorapis.portal.chatbot.data.local.model;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;

public class BotMessage {
    public final static User mBot = new User(Constants.BOT_ID, "M");

    private Message.Builder mBotMessage = new Message.Builder()
            .setUser(mBot)
            .setPosition(Message.Position.LEFT);

    private InteractType interactType = InteractType.NONE;

    public BotMessage(InteractType interactType) {
        this.interactType = interactType;
    }

    public BotMessage() {
    }

    public Message getMessage() {
        return mBotMessage.build();
    }

    public InteractType getInteractType() {
        return interactType;
    }

    public void setText(String text) {
        mBotMessage.setText(text);
    }

    public void setChoices(Choices choices) {
        mBotMessage.setChoices(choices);
    }

    public enum InteractType {
        HELLO,
        FOR_WHOM,
        GENDER,
        AGE,
        CONVERSATION,
        SYMPTOMS,
        QUESTIONS,
        QUESTIONS_SECONDARY,
        QUESTIONS_PRIMARY,
        CAUSES,
        NONE
    }
}
