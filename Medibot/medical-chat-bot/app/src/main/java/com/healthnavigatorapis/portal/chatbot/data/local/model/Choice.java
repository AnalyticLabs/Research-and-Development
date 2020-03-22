package com.healthnavigatorapis.portal.chatbot.data.local.model;

public class Choice {
    private int choiceId;
    private String value;
    private boolean isSelected;

    public Choice(int choiceId, String value) {
        this.choiceId = choiceId;
        this.value = value;
    }

    public Choice(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public int getChoiceId() {
        return choiceId;
    }

    public boolean isSelected() {
        return isSelected;
    }

    public void setSelected(boolean selected) {
        isSelected = selected;
    }

}
