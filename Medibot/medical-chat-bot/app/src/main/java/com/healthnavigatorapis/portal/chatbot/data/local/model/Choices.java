package com.healthnavigatorapis.portal.chatbot.data.local.model;

import java.util.List;

public class Choices {
    private ChoiceType type;
    private List<Choice> choiceList;

    public Choices(ChoiceType type, List<Choice> choiceList) {
        this.type = type;
        this.choiceList = choiceList;
    }

    public ChoiceType getType() {
        return type;
    }

    public void setType(ChoiceType type) {
        this.type = type;
    }

    public List<Choice> getChoiceList() {
        return choiceList;
    }

    public enum ChoiceType {
        SINGLE,
        MULTIPLE
    }
}
