package com.healthnavigatorapis.portal.chatbot.ui.chat;

import android.app.Application;
import android.text.TextUtils;
import android.widget.Toast;

import com.healthnavigatorapis.portal.chatbot.App;
import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.arch.BaseViewModel;
import com.healthnavigatorapis.portal.chatbot.arch.SingleLiveEvent;
import com.healthnavigatorapis.portal.chatbot.data.DataRepository;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;
import com.healthnavigatorapis.portal.chatbot.data.local.model.BotMessage;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choice;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Choices;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Message;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Question;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.QuestionPrimary;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.RequestData;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Symptom;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.TriageScore;
import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;
import com.healthnavigatorapis.portal.chatbot.util.BotLogic;
import com.healthnavigatorapis.portal.chatbot.util.Utils;

import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import androidx.annotation.NonNull;
import androidx.databinding.ObservableBoolean;
import androidx.lifecycle.MutableLiveData;
import io.reactivex.Completable;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;

public class ChatViewModel extends BaseViewModel implements BotLogic.BotListener {

    private final MutableLiveData<Message> mMessageLiveData = new MutableLiveData<>();
    private final MutableLiveData<List<Message>> mMessageList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> mIsShowInput = new MutableLiveData<>();
    private final RequestData mRequestData = new RequestData();
    private final ObservableBoolean mIsMultiChoices = new ObservableBoolean(false);
    private final SingleLiveEvent<Void> mVoiceRecognition = new SingleLiveEvent<>();
    private final SingleLiveEvent<Void> mRefreshPressed = new SingleLiveEvent<>();
    private final SingleLiveEvent<String> mError = new SingleLiveEvent<>();
    private final MutableLiveData<Boolean> mIsChoices = new MutableLiveData<>();

    private DataRepository mRepository;
    private BotLogic botLogic;
    private MutableLiveData<String> textField = new MutableLiveData<>();
    private User user;
    private BotMessage.InteractType mBotInteractType;
    private CompositeDisposable compositeDisposable = new CompositeDisposable();

    public ChatViewModel(@NonNull Application application) {
        super(application);
        mRepository = ((App) getApplication()).getRepository();
        botLogic = new BotLogic(getApplication());
        botLogic.setListener(this);
        mIsShowInput.setValue(true);
        onBotStart();
    }

    private void setRequestData() {
        mRequestData.clear();
        mRequestData.setAge(user.geAgeInDays());
        mRequestData.setGender(user.getGender());
    }

    public void onSendMessage() {
        onSendMessage(textField.getValue(), true);
    }

    public MutableLiveData<Message> getMessageLiveData() {
        return mMessageLiveData;
    }

    public void setMessageLiveData(Message message) {
        mMessageLiveData.setValue(message);
    }

    public void postMessageLiveData(Message message) {
        mMessageLiveData.postValue(message);
    }

    public void onArrowClicked() {
        mIsShowInput.setValue(!mIsShowInput.getValue());
    }

    public MutableLiveData<Boolean> getIsShowInput() {
        return mIsShowInput;
    }

    public MutableLiveData<String> getTextField() {
        return textField;
    }

    public ObservableBoolean getIsMultiChoices() {
        return mIsMultiChoices;
    }

    public MutableLiveData<Boolean> getIsChoices() {
        return mIsChoices;
    }

    public SingleLiveEvent<Void> getVoiceRecognition() {
        return mVoiceRecognition;
    }

    public SingleLiveEvent<Void> getRefreshPressed() {
        return mRefreshPressed;
    }

    public MutableLiveData<List<Message>> getMessageList() {
        return mMessageList;
    }

    public SingleLiveEvent<String> getError() {
        return mError;
    }

    public void onBackPressed() {
        setNavigation(FragmentNavigator.Navigation.Back);
    }

    public void onRefreshPressed() {
        mRefreshPressed.call();
        mIsMultiChoices.set(false);
        mIsShowInput.postValue(true);
        choicesShow(false);
        onBotStart();
    }

    public void onHelpPressed() {
        Toast.makeText(getApplication(), R.string.is_not_available, Toast.LENGTH_SHORT).show();
    }

    public void onVoiceRecognition() {
        mVoiceRecognition.call();
    }

    public void onBotStart() {
        user = mRepository.getUser(Constants.USER_ID).blockingGet();
        setRequestData();
        BotMessage botMessage = botLogic.onStartBot(user);
        setMessageLiveData(botMessage.getMessage());
        mBotInteractType = botMessage.getInteractType();
    }

    public void selectedChoice(List<Choice> choices) {
        Disposable disposable = Completable.timer(Constants.DELAY_MILLI + 100, TimeUnit.MILLISECONDS, Schedulers.io())
                .subscribe(() -> {
                    int choiceId = -1;
                    if (!choices.isEmpty()) {
                        choiceId = choices.get(0).getChoiceId();
                    }
                    switch (mBotInteractType) {
                        case GENDER: {
                            user.setGender(choices.get(0).getValue());
                            mRequestData.setGender(choices.get(0).getValue());
                            break;
                        }
                        case SYMPTOMS: {
                            mRequestData.setConceptId(choices.get(0).getValue(), choices.get(0).getChoiceId());
                            break;
                        }
                        case QUESTIONS: {
                            outputChoices(choiceId, choices, BotMessage.InteractType.QUESTIONS);
                            return;
                        }
                        case QUESTIONS_SECONDARY:
                            outputChoices(choiceId, choices, BotMessage.InteractType.QUESTIONS_SECONDARY);
                            return;
                        case QUESTIONS_PRIMARY:
                            outputChoices(choiceId, choices, BotMessage.InteractType.QUESTIONS_PRIMARY);
                            return;
                    }
                    outputSelectedChoices(choices);
                });
        compositeDisposable.add(disposable);
    }

    private void outputChoices(int choiceId, List<Choice> choices, BotMessage.InteractType interactType) {
        if (interactType == BotMessage.InteractType.QUESTIONS_PRIMARY) {
            if (!choices.isEmpty()) {
                for (Choice choice : choices) {
                    mRequestData.addConceptPresent(choice.getChoiceId());
                }
                choiceId = -1;
            } else {
                onSendMessage(Constants.MESSAGE_NOT_CHOOSE, false);
            }
        }
        outputSelectedChoices(choices);
        consistentQuestions(choiceId, interactType);
    }

    private void outputSelectedChoices(List<Choice> choices) {
        if (choices.size() == 1) {
            onSendMessage(choices.get(0).getValue(), false);
        } else {
            List<Message> messages = new ArrayList<>();
            for (Choice choice : choices) {
                messages.add(new Message.Builder()
                        .setUser(user)
                        .setPosition(Message.Position.RIGHT)
                        .setText(choice.getValue())
                        .build());
            }
            mMessageList.postValue(messages);
        }
    }

    public void onSendMessage(String value, boolean isMainThread) {
        if (!TextUtils.isEmpty(value)) {
            if (isMainThread) {
                setMessageLiveData(new Message.Builder()
                        .setUser(user)
                        .setPosition(Message.Position.RIGHT)
                        .setText(value)
                        .build());
                textField.setValue("");
            } else {
                postMessageLiveData(new Message.Builder()
                        .setUser(user)
                        .setPosition(Message.Position.RIGHT)
                        .setText(value)
                        .build());
            }

            if (parserInteractions(mBotInteractType, value)) {
                onInteractWithBot(botLogic.interactionWithBot(user, value, mBotInteractType), isMainThread);
            }
        }
    }

    private void onInteractWithBot(BotMessage botMessage, boolean isMainThread) {
        if (botMessage != null) {
            Message message = botMessage.getMessage();
            if (isMainThread) {
                setMessageLiveData(message);
            } else {
                Disposable disposable = Completable.timer(Constants.DELAY_MILLI, TimeUnit.MILLISECONDS, Schedulers.io())
                        .subscribe(() -> {
                            postMessageLiveData(message);
                        });
                compositeDisposable.add(disposable);
            }
            mBotInteractType = botMessage.getInteractType();
        }
    }

    private boolean parserInteractions(BotMessage.InteractType interactType, String value) {
        switch (interactType) {
            case AGE:
                int age = Utils.ageParse(value);
                if (age != -1) {
                    user.setAge(age);
                    mRequestData.setAgeInDays(age);
                    return true;
                } else {
                    setMessageLiveData(botLogic.invalidAge().getMessage());
                    return false;
                }
            case CONVERSATION:
                if (!TextUtils.isEmpty(value)) {
                    getSymptoms(value);
                }
                return false;
            case SYMPTOMS:
                getQuestion();
                return false;
        }
        return true;
    }

    private void getSymptoms(String value) {
        Disposable disposable = mRepository.findCCC(user.geAgeInDays(), user.getGender(), value)
                .subscribe(this::prepareSymptoms);
        compositeDisposable.add(disposable);
    }

    private void getQuestion() {
        Disposable disposable = mRepository.getQuestionsPrimary(mRequestData.getConceptId(), mRequestData.getAgeInDays(), mRequestData.getGender())
                .subscribe((questions, throwable) -> prepareQuestions(questions, BotMessage.InteractType.QUESTIONS, throwable));
        compositeDisposable.add(disposable);
    }

    private void getSecondaryQuestion() {
        Integer[] secondaryPresent = mRequestData.getSecondaryCCCPresent();
        if (secondaryPresent != null) {
            Disposable disposable = mRepository.getQuestionsSecondary(mRequestData.getConceptId(), mRequestData.getAgeInDays(), mRequestData.getGender(), secondaryPresent)
                    .subscribe((questions, throwable) -> prepareQuestions(questions, BotMessage.InteractType.QUESTIONS_SECONDARY, throwable));
            compositeDisposable.add(disposable);
        } else {
            getQuestionsPrimaryCCC();
        }
    }

    private void getQuestionsPrimaryCCC() {
        Disposable disposable = mRepository.getQuestionsPrimaryCCC(mRequestData.getConceptId(), mRequestData.getAgeInDays(), mRequestData.getGender(), mRequestData.getConceptPresent())
                .subscribe(this::prepareQuestionPrimary);
        compositeDisposable.add(disposable);
        mIsMultiChoices.set(true);
        mIsShowInput.postValue(false);
    }

    private void getTriageScore() {
        Disposable disposable = mRepository.getTriageScore(mRequestData.getConceptId(), mRequestData.getAgeInDays(), mRequestData.getGender(), mRequestData.getConceptPresent())
                .subscribe(this::prepareTriageScore);
        compositeDisposable.add(disposable);
    }

    private void getCauses() {
        mIsMultiChoices.set(false);
        mIsShowInput.postValue(false);
        Disposable disposable = mRepository.getCauses(mRequestData.getConceptId(), mRequestData.getAgeInDays(), mRequestData.getGender(), mRequestData.getConceptPresent())
                .subscribe(this::prepareCauses);
        compositeDisposable.add(disposable);
    }

    private boolean onErrorParse(Throwable throwable) {
        if (throwable == null) {
            return true;
        }
        if (throwable instanceof UnknownHostException) {
            mError.postValue("No internet connection");
        }
        return false;
    }

    private void prepareCauses(List<Cause> causes, Throwable throwable) {
        if (onErrorParse(throwable)) {
            List<Cause> causeList = new ArrayList<>();
            int size = Constants.MAX_CAUSES_SIZE;
            if (!causes.isEmpty() && causes.size() <= size) {
                size = causes.size();
            }
            for (int i = 0; i < size; i++) {
                causeList.add(causes.get(i));
            }

            mMessageLiveData.postValue(new Message.Builder()
                    .setText("Top Causes")
                    .setCause(causeList)
                    .setPosition(Message.Position.CENTER)
                    .build());
        }
    }

    private void prepareTriageScore(List<TriageScore> triageScores, Throwable throwable) {
        if (onErrorParse(throwable)) {
            TriageScore triageScore = triageScores.get(0);
            List<Message> messages = new ArrayList<>();

            Message message = new Message.Builder()
                    .setPosition(Message.Position.LEFT)
                    .setUser(BotMessage.mBot)
                    .setText(triageScore.getRecommendedCareLocalized())
                    .build();
            messages.add(message);

            Message getInfo = new Message.Builder()
                    .setPosition(Message.Position.LEFT)
                    .setUser(BotMessage.mBot)
                    .setText(getInfo())
                    .build();

            Message diagnosis = new Message.Builder()
                    .setPosition(Message.Position.LEFT)
                    .setUser(BotMessage.mBot)
                    .setText("<b>Diagnosis Details</b>")
                    .build();

            messages.add(getInfo);
            messages.add(diagnosis);
            mMessageList.postValue(messages);
        }
        getCauses();
    }

    private String getInfo() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("<b>Patient Details:</b> <br/>")
                .append("Age & sex: ")
                .append(user.getAge())
                .append(" years adult ")
                .append(user.getFullGender())
                .append("<br/>")
                .append("Primary Complaint: ")
                .append(mRequestData.getPrimaryComplaint())
                .append("<br/>")
                .append("Secondary Complaint: ");
        if (mRequestData.getSecondaryComplaint().isEmpty()) {
            stringBuilder.append("N.A.")
                    .append("<br/>");
        } else {
            for (int i = 0; i < mRequestData.getSecondaryComplaint().size(); i++) {
                stringBuilder.append(mRequestData.getSecondaryComplaint().get(i));
                if (i < mRequestData.getSecondaryComplaint().size() - 1) {
                    stringBuilder.append(", ");
                } else {
                    stringBuilder.append(".<br/>");
                }
            }
        }
        stringBuilder.append("Medical History: N.A.");

        return stringBuilder.toString();
    }

    private void prepareQuestions(List<Question> questions, BotMessage.InteractType interactType, Throwable throwable) {
        if (onErrorParse(throwable)) {
            botLogic.prepareQuestionsMessage(questions, interactType);
            onInteractWithBot(botLogic.getConsistentMessage(), false);
        }
    }

    private void prepareQuestionPrimary(List<QuestionPrimary> questions, Throwable throwable) {
        if (onErrorParse(throwable)) {
            botLogic.preparePrimaryMessage(questions);
            onInteractWithBot(botLogic.getConsistentMessage(), false);
        }
    }

    private void prepareSymptoms(List<Symptom> symptoms, Throwable throwable) {
        if (onErrorParse(throwable)) {
            if (!symptoms.isEmpty() && botLogic.isSymptoms(symptoms)) {
                for (BotMessage botMessage : botLogic.prepareSymptomsMessage(user, symptoms)) {
                    if (botMessage.getInteractType() == BotMessage.InteractType.SYMPTOMS) {
                        Choices choices = botMessage.getMessage().getChoices();
                        if (choices != null) {
                            for (Choice choice : choices.getChoiceList()) {
                                mRequestData.addConceptPresent(choice.getChoiceId());
                                mRequestData.addSecondaryCCCPresent(choice.getValue(), choice.getChoiceId());
                            }
                        }
                    }
                    onInteractWithBot(botMessage, false);
                }
            } else {
                BotMessage botMessage = new BotMessage(BotMessage.InteractType.CONVERSATION);
                botMessage.setText(getApplication().getResources().getString(R.string.bot_conservation_error));
                postMessageLiveData(botMessage.getMessage());
            }
        }
    }

    private void consistentQuestions(int id, BotMessage.InteractType interactType) {
        if (id != -1) {
            mRequestData.addConceptPresent(id);
        }
        BotMessage consistentMessage = botLogic.getConsistentMessage();
        if (consistentMessage != null) {
            onInteractWithBot(consistentMessage, false);
        } else {
            if (interactType == BotMessage.InteractType.QUESTIONS) {
                getSecondaryQuestion();
            } else if (interactType == BotMessage.InteractType.QUESTIONS_SECONDARY) {
                getQuestionsPrimaryCCC();
            } else if (interactType == BotMessage.InteractType.QUESTIONS_PRIMARY) {
                getTriageScore();
            }
        }
    }

    @Override
    public void userChanged(int id) {
        user.setId(id);
    }

    @Override
    public void choicesShow(boolean isShow) {
        mIsChoices.postValue(isShow);
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        if (compositeDisposable != null && !compositeDisposable.isDisposed()) {
            compositeDisposable.dispose();
        }
    }
}
