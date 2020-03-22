package com.healthnavigatorapis.portal.chatbot.util;

import android.content.Context;
import android.text.TextUtils;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.local.model.BotMessage;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;
import com.healthnavigatorapis.portal.chatbot.data.local.model.IChatUser;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Question;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.QuestionPrimary;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Symptom;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;

public class BotLogic {
    private static final String SYMPTOM_TYPE = "symptom";
    private final Context mContext;
    private BotListener mListener;
    private List<BotMessage> tempBotMessage;

    public BotLogic(Context context) {
        mContext = context;
    }

    public void setListener(BotListener listener) {
        mListener = listener;
    }

    public BotMessage interactionWithBot(IChatUser chatUser, String message, BotMessage.InteractType interactType) {
        switch (interactType) {
            case HELLO:
                if (message.toLowerCase().contains("symptoms")) {
                    return parseForWho(message, chatUser);
                } else {
                    return didNotUnderstoodSymptoms();
                }
            case FOR_WHOM:
                return parseForWho(message, chatUser);
            case GENDER:
                return howOld(chatUser);
            case AGE:
                return conversationSymptoms(chatUser);
        }
        return null;
    }

    private BotMessage parseForWho(String message, IChatUser chatUser) {
        if (message.toLowerCase().contains("someone") || message.toLowerCase().contains("somebody")) {
            if (mListener != null) {
                mListener.userChanged(Constants.SOMEONE_ID);
            }
            return whatGender(chatUser);
        } else if (message.toLowerCase().contains("me") || message.toLowerCase().contains("my") || message.toLowerCase().contains("mine")) {
            if (chatUser.getGender().equals("B")) {
                return whatGender(chatUser);
            } else if (chatUser.getAge() == 0) {
                return howOld(chatUser);
            } else {
                return conversationSymptoms(chatUser);
            }
        } else {
            return forWho();
        }
    }

    public BotMessage onStartBot(IChatUser chatUser) {
        BotMessage message = new BotMessage(BotMessage.InteractType.HELLO);
        if (chatUser != null && !TextUtils.isEmpty(chatUser.getName())) {
            message.setText(mContext.getResources()
                    .getString(R.string.bot_hello, chatUser.getName()));
        } else {
            message.setText(mContext.getResources()
                    .getString(R.string.bot_hello, "User"));
        }
        return message;
    }

    private BotMessage forWho() {
        BotMessage message = new BotMessage(BotMessage.InteractType.FOR_WHOM);
        message.setText(mContext.getResources().getString(R.string.bot_for_who));
        return message;
    }

    private BotMessage whatGender(IChatUser chatUser) {
        if (mListener != null) {
            mListener.choicesShow(true);
        }
        BotMessage message = new BotMessage(BotMessage.InteractType.GENDER);

        ArrayList<Choice> choiceList = new ArrayList<Choice>() {{
            add(new Choice("Male"));
            add(new Choice("Female"));
        }};

        addMessage(message, chatUser.getType(),
                mContext.getResources().getString(R.string.bot_gender_user),
                mContext.getResources().getString(R.string.bot_gender_someone));

        message.setChoices(new Choices(Choices.ChoiceType.SINGLE, choiceList));
        return message;
    }

    private BotMessage howOld(IChatUser chatUser) {
        if (mListener != null) {
            mListener.choicesShow(false);
        }
        BotMessage message = new BotMessage(BotMessage.InteractType.AGE);

        addMessage(message, chatUser.getType(),
                mContext.getResources().getString(R.string.bot_old_user),
                mContext.getResources().getString(R.string.bot_old_someone, parseGender(chatUser.getGender())));

        return message;
    }

    private BotMessage conversationSymptoms(IChatUser chatUser) {
        BotMessage message = new BotMessage(BotMessage.InteractType.CONVERSATION);
        addMessage(message, chatUser.getType(),
                mContext.getResources().getString(R.string.bot_conservation, "you are"),
                mContext.getResources().getString(R.string.bot_conservation, parseGender(chatUser.getGender(), false, true, false)));
        return message;
    }

    public BotMessage didNotUnderstoodSymptoms() {
        BotMessage message = new BotMessage(BotMessage.InteractType.HELLO);
        message.setText(mContext.getResources().getString(R.string.bot_conservation_did_not));
        return message;
    }

    public BotMessage invalidAge() {
        BotMessage message = new BotMessage(BotMessage.InteractType.AGE);
        message.setText("Please input the correct age in years.");
        return message;
    }

    private String prepareMessage(ArrayList<Choice> choice) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < choice.size(); i++) {
            if (i != choice.size() - 1) {
                builder.append(choice.get(i).getValue()).append(" and ");
            } else {
                builder.append(choice.get(i).getValue());
            }
        }
        return builder.toString();
    }

    private String parseGender(String gender, boolean secondForm, boolean thirdForm, boolean has) {
        if (gender.toLowerCase().contains("f")) {
            if (secondForm) {
                return "her";
            } else if (thirdForm) {
                return "she is";
            } else if (has) {
                return "she has";
            } else {
                return "she";
            }
        } else if (gender.toLowerCase().contains("m")) {
            if (secondForm) {
                return "him";
            } else if (thirdForm) {
                return "he is";
            } else if (has) {
                return "he has";
            } else {
                return "he";
            }
        }
        return "he";
    }

    private String parseGender(String gender) {
        return parseGender(gender, false, false, false);
    }

    private void addMessage(BotMessage message, IChatUser.UserType userType, String mainUser, String somebody) {
        switch (userType) {
            case MAIN_USER:
                message.setText(mainUser);
                break;
            case SOMEONE_ELSE:
                message.setText(somebody);
                break;
            default:
                break;
        }
    }

    public List<BotMessage> prepareSymptomsMessage(IChatUser chatUser, List<Symptom> symptomList) {
        if (mListener != null) {
            mListener.choicesShow(true);
        }

        List<BotMessage> botMessages = new ArrayList<>();

        BotMessage message = new BotMessage(BotMessage.InteractType.SYMPTOMS);
        ArrayList<Choice> choiceList = new ArrayList<>();
        for (Symptom symptom : symptomList) {
            if (isSymptom(symptom)) {
                choiceList.add(new Choice(symptom.getConceptId(), symptom.getTitle()));
            }
        }
        message.setChoices(new Choices(Choices.ChoiceType.SINGLE, choiceList));
        addMessage(message, chatUser.getType(),
                mContext.getResources().getString(R.string.bot_symptoms_second_message, "you"),
                mContext.getResources().getString(R.string.bot_symptoms_second_message, parseGender(chatUser.getGender(), true, false, false)));
        String choices = prepareMessage(choiceList);
        BotMessage mainMessage = new BotMessage();
        addMessage(mainMessage, chatUser.getType(),
                mContext.getResources().getString(R.string.bot_symptoms_list, "your", choices),
                mContext.getResources().getString(R.string.bot_symptoms_list, parseGender(chatUser.getGender(), false, false, true), choices));

        botMessages.add(mainMessage);
        botMessages.add(message);

        return botMessages;
    }

    public void prepareQuestionsMessage(@NonNull List<Question> questions, BotMessage.InteractType interactType) {
        if (mListener != null) {
            mListener.choicesShow(true);
        }

        List<BotMessage> botMessages = new ArrayList<>();

        BotMessage message = new BotMessage(interactType);
        ArrayList<Choice> choiceList = new ArrayList<>();
        String temp = questions.get(0).getOPQRSTGroupQuestionPLLocalized();
        for (int i = 0; i < questions.size(); i++) {
            Question question = questions.get(i);
            if (temp.equals(question.getOPQRSTGroupQuestionPLLocalized())) {
                if (botMessages.isEmpty()) {
                    message.setText(temp);
                    botMessages.add(message);
                }
            } else {
                temp = question.getOPQRSTGroupQuestionPLLocalized();
                message.setChoices(new Choices(Choices.ChoiceType.SINGLE, choiceList));
                choiceList = new ArrayList<>();
                message = new BotMessage(interactType);
                message.setText(temp);
                botMessages.add(message);
            }
            choiceList.add(new Choice(question.getConceptId(), question.getTitleLocalized()));
            if (i == questions.size() - 1) {
                message.setChoices(new Choices(Choices.ChoiceType.SINGLE, choiceList));
            }

        }
        tempBotMessage = botMessages;
    }

    public void preparePrimaryMessage(@NonNull List<QuestionPrimary> questions) {
        if (mListener != null) {
            mListener.choicesShow(true);
        }

        List<BotMessage> botMessages = new ArrayList<>();

        BotMessage message = new BotMessage(BotMessage.InteractType.QUESTIONS_PRIMARY);
        ArrayList<Choice> choiceList = new ArrayList<>();
        String temp = questions.get(0).getElementPLQuestionLocalized();
        for (int i = 0; i < questions.size(); i++) {
            QuestionPrimary question = questions.get(i);
            if (temp.equals(question.getElementPLQuestionLocalized())) {
                if (botMessages.isEmpty()) {
                    message.setText(temp);
                    botMessages.add(message);
                }
            } else {
                temp = question.getElementPLQuestionLocalized();
                message.setChoices(new Choices(Choices.ChoiceType.MULTIPLE, choiceList));
                choiceList = new ArrayList<>();
                message = new BotMessage(BotMessage.InteractType.QUESTIONS_PRIMARY);
                message.setText(temp);
                botMessages.add(message);
            }
            choiceList.add(new Choice(question.getConceptId(), question.getTitlePlainLanguage()));
            if (i == questions.size() - 1) {
                message.setChoices(new Choices(Choices.ChoiceType.MULTIPLE, choiceList));
            }

        }
        tempBotMessage = botMessages;
    }

    public BotMessage getConsistentMessage() {
        BotMessage message = null;
        if (!tempBotMessage.isEmpty()) {
            message = tempBotMessage.get(0);
            tempBotMessage.remove(0);
        }
        return message;
    }

    public boolean isSymptoms(List<Symptom> symptoms) {
        for (Symptom symptom : symptoms) {
            if (isSymptom(symptom)) {
                return true;
            }
        }
        return false;
    }

    private boolean isSymptom(Symptom symptom) {
        return symptom.getType().equalsIgnoreCase(SYMPTOM_TYPE);
    }

    public interface BotListener {
        void userChanged(int id);

        void choicesShow(boolean isShow);
    }
}
