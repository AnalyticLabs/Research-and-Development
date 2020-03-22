package com.healthnavigatorapis.portal.chatbot.data.remote.service;

import com.healthnavigatorapis.portal.chatbot.Constants;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Cause;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Question;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.QuestionPrimary;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.Symptom;
import com.healthnavigatorapis.portal.chatbot.data.remote.model.TriageScore;

import java.util.List;

import io.reactivex.Single;
import retrofit2.http.GET;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface HealthService {
    @GET(Constants.FIND_CCC)
    Single<List<Symptom>> findCCC(@Query("ageInDays") long ageInDays,
                                  @Query("gender") String gender,
                                  @Query("freeTextChiefComplaints") String text);

    @GET(Constants.TELL_US_MORE_PRIMARY_CCC_OPQRST + "/{id}")
    Single<List<Question>> getQuestionsPrimary(@Path("id") int id,
                                               @Query("ageInDays") long ageInDays,
                                               @Query("gender") String gender);

    @GET(Constants.TELL_US_MORE_SECONDARY_CCC_OPQRST + "/{id}")
    Single<List<Question>> getQuestionsSecondary(@Path("id") int id,
                                                 @Query("ageInDays") long ageInDays,
                                                 @Query("gender") String gender,
                                                 @Query("secondaryCCCPresent") Integer... secondaryCCCPresent);

    @GET(Constants.TELL_US_MORE_PRIMARY_CCC + "/{id}")
    Single<List<QuestionPrimary>> getQuestionsPrimaryCCC(@Path("id") int id,
                                                         @Query("ageInDays") long ageInDays,
                                                         @Query("gender") String gender,
                                                         @Query("conceptPresent") Integer... conceptPresent);

    @GET(Constants.GET_CAUSES + "/{id}")
    Single<List<Cause>> getCauses(@Path("id") int id,
                                  @Query("ageInDays") long ageInDays,
                                  @Query("gender") String gender,
                                  @Query("filterCauses") String filterCauses,
                                  @Query("conceptPresent") Integer... conceptPresent);


    @GET(Constants.CALCULATE_TRIAGE_SCORE + "/{id}")
    Single<List<TriageScore>> getCalculateTriageScore(@Path("id") int id,
                                                      @Query("ageInDays") long ageInDays,
                                                      @Query("gender") String gender,
                                                      @Query("conceptPresent") Integer... conceptPresent);
}
