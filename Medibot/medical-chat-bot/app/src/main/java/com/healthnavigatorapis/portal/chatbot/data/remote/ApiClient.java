package com.healthnavigatorapis.portal.chatbot.data.remote;

import com.healthnavigatorapis.portal.chatbot.Constants;

import androidx.annotation.NonNull;
import okhttp3.Credentials;
import okhttp3.Headers;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

public class ApiClient {
    private static final String AUTH_TOKEN = "Basic Mjc1YmZmNDctNTg3YS00Y2UwLWJkZTMtMTE2MTU5ZTM2YjQxOjY3YWY1MGI1LTQ1NTktNDk5ZC1iMmQ4LTI5NDlkNzE5YTAzYQ==";
    private static Retrofit mRetrofit = null;
    private static String authToken;

    public static Retrofit initAuthClient(@NonNull String login, @NonNull String password) {
        authToken = Credentials.basic(login, password);

        OkHttpClient httpClient = new OkHttpClient.Builder().addInterceptor(chain -> {
            Request request = chain.request();
            Headers headers = request.headers()
                    .newBuilder()
                    .add("Authorization", authToken)
                    .build();
            request = request.newBuilder()
                    .headers(headers)
                    .build();
            return chain.proceed(request);
        }).build();

        mRetrofit = new Retrofit.Builder()
                .baseUrl(Constants.BASE_URL)
                .client(httpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
                .build();

        return mRetrofit;
    }

    public static Retrofit getClient() {
        if (mRetrofit == null) {
            OkHttpClient httpClient = new OkHttpClient.Builder().addInterceptor(chain -> {
                Request request = chain.request();
                Headers headers = request.headers()
                        .newBuilder()
                        .add("Authorization", AUTH_TOKEN)
                        .build();
                request = request.newBuilder()
                        .headers(headers)
                        .build();
                return chain.proceed(request);
            }).build();

            mRetrofit = new Retrofit.Builder()
                    .baseUrl(Constants.BASE_URL)
                    .client(httpClient)
                    .addConverterFactory(GsonConverterFactory.create())
                    .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
                    .build();
        }
        return mRetrofit;
    }
}
