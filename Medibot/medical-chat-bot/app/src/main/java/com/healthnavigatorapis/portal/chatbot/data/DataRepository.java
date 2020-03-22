package com.healthnavigatorapis.portal.chatbot.data;

import com.healthnavigatorapis.portal.chatbot.data.local.AppDatabase;
import com.healthnavigatorapis.portal.chatbot.data.local.entity.User;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Question;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.QuestionPrimary;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Symptom;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.TriageScore;
import com.healthnavigatorapis.portal.chatbot.data.remote.service.HealthService;

import java.util.List;

import io.reactivex.Observable;
import io.reactivex.Single;
import io.reactivex.schedulers.Schedulers;

public class DataRepository {

    private static DataRepository mInstance;

    private final AppDatabase mDatabase;
    private final HealthService mService;

    private DataRepository(AppDatabase database, HealthService service) {
        mDatabase = database;
        mService = service;
    }

    public static DataRepository getInstance(final AppDatabase database, final HealthService service) {
        if (mInstance == null) {
            synchronized (DataRepository.class) {
                if (mInstance == null) {
                    mInstance = new DataRepository(database, service);
                }
            }
        }
        return mInstance;
    }

    public Single<User> getUser(int userId) {
        return mDatabase.getUserDao()
                .getUser(userId)
                .subscribeOn(Schedulers.io());
    }

    public Observable<Long> insertUser(User user) {
        return Observable.fromCallable(() -> mDatabase.getUserDao().insert(user));
    }

    public Single<List<Symptom>> findCCC(long ageInDays, String gender, String text) {
        return mService.findCCC(ageInDays, gender, text)
                .subscribeOn(Schedulers.io());
    }

    public Single<List<Question>> getQuestionsPrimary(int conceptId, long ageInDays, String gender) {
        return mService.getQuestionsPrimary(conceptId, ageInDays, gender)
                .subscribeOn(Schedulers.io());
    }

    public Single<List<Question>> getQuestionsSecondary(int conceptId, long ageInDays, String gender, Integer[] secondaryCCCPresent) {
        return mService.getQuestionsSecondary(conceptId, ageInDays, gender, secondaryCCCPresent)
                .subscribeOn(Schedulers.io());
    }

    public Single<List<QuestionPrimary>> getQuestionsPrimaryCCC(int conceptId, long ageInDays, String gender, Integer[] conceptPresent) {
        return mService.getQuestionsPrimaryCCC(conceptId, ageInDays, gender, conceptPresent)
                .subscribeOn(Schedulers.io());
    }

    public Single<List<Cause>> getCauses(int conceptId, long ageInDays, String gender, Integer[] conceptPresent) {
        return mService.getCauses(conceptId, ageInDays, gender, "All", conceptPresent)
                .subscribeOn(Schedulers.io());
    }

    public Single<List<TriageScore>> getTriageScore(int conceptId, long ageInDays, String gender, Integer[] conceptPresent) {
        return mService.getCalculateTriageScore(conceptId, ageInDays, gender, conceptPresent)
                .subscribeOn(Schedulers.io());
    }
}
