package com.healthnavigatorapis.portal.chatbot.data.local.model;

import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;

import java.util.List;

public class Message {
    private int id;
    private IChatUser user;
    private Position position;
    private String text;
    private Choices choices;
    private List<Cause> causes;
    private boolean isChoice;

    public boolean isChoice() {
        return isChoice;
    }

    public Position getPosition() {
        return position;
    }

    public String getText() {
        return text;
    }

    public IChatUser getUser() {
        return user;
    }

    public Choices getChoices() {
        return choices;
    }

    public int getId() {
        return id;
    }

    public List<Cause> getCauses() {
        return causes;
    }

    public enum Position {
        RIGHT,
        LEFT,
        CENTER
    }

    public static class Builder {
        private Message message = new Message();

        public Builder setUser(IChatUser user) {
            message.user = user;
            return this;
        }

        public Builder setPosition(Position position) {
            message.position = position;
            return this;
        }

        public Builder setText(String text) {
            message.text = text;
            return this;
        }

        public Builder setChoices(Choices choices) {
            message.choices = choices;
            message.isChoice = true;
            return this;
        }

        public Builder setCause(List<Cause> cause) {
            message.causes = cause;
            return this;
        }

        public Builder setId(int id) {
            message.id = id;
            return this;
        }

        public Message build() {
            return message;
        }
    }
}