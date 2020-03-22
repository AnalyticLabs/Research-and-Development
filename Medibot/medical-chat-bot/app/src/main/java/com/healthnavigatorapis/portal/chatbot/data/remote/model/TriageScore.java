package com.healthnavigatorapis.portal.chatbot.data.remote.model;

import com.google.gson.annotations.SerializedName;

public class TriageScore {

    @SerializedName("TriageRatingMainNumeric")
    private int triageRatingMainNumeric;

    @SerializedName("TriageRatingNumeric")
    private int triageRatingNumeric;

    @SerializedName("RecommendedCare_LocationCode")
    private String recommendedCareLocationCode;

    @SerializedName("ResultStatus")
    private int resultStatus;

    @SerializedName("RecommendedCare")
    private String recommendedCare;

    @SerializedName("ResultStatusDescription")
    private String resultStatusDescription;

    @SerializedName("RecommendedCare_LocationTiming_Localized")
    private String recommendedCareLocationTimingLocalized;

    @SerializedName("TriageRatingColorImgURL")
    private String triageRatingColorImgURL;

    @SerializedName("TriageRatingCallDoctorTitleLocalized")
    private String triageRatingCallDoctorTitleLocalized;

    @SerializedName("RecommendedCare_LocationDesc")
    private String recommendedCareLocationDesc;

    @SerializedName("RecommendedCare_LocationDesc_Localized")
    private String recommendedCareLocationDescLocalized;

    @SerializedName("RecommendedCare_LocationTiming")
    private String recommendedCareLocationTiming;

    @SerializedName("PrimaryCCC")
    private int primaryCCC;

    @SerializedName("TriageRatingCallDoctorTitle")
    private String triageRatingCallDoctorTitle;

    @SerializedName("RecommendedCareLocalized")
    private String recommendedCareLocalized;

    @SerializedName("RecommendedCare_LocationCost4")
    private int recommendedCareLocationCost4;

    @SerializedName("TriageRatingSeekCareTitle")
    private String triageRatingSeekCareTitle;

    @SerializedName("TriageRatingSeekCareTitleLocalized")
    private String triageRatingSeekCareTitleLocalized;

    public int getTriageRatingMainNumeric() {
        return triageRatingMainNumeric;
    }

    public void setTriageRatingMainNumeric(int triageRatingMainNumeric) {
        this.triageRatingMainNumeric = triageRatingMainNumeric;
    }

    public int getTriageRatingNumeric() {
        return triageRatingNumeric;
    }

    public void setTriageRatingNumeric(int triageRatingNumeric) {
        this.triageRatingNumeric = triageRatingNumeric;
    }

    public String getRecommendedCareLocationCode() {
        return recommendedCareLocationCode;
    }

    public void setRecommendedCareLocationCode(String recommendedCareLocationCode) {
        this.recommendedCareLocationCode = recommendedCareLocationCode;
    }

    public int getResultStatus() {
        return resultStatus;
    }

    public void setResultStatus(int resultStatus) {
        this.resultStatus = resultStatus;
    }

    public String getRecommendedCare() {
        return recommendedCare;
    }

    public void setRecommendedCare(String recommendedCare) {
        this.recommendedCare = recommendedCare;
    }

    public String getResultStatusDescription() {
        return resultStatusDescription;
    }

    public void setResultStatusDescription(String resultStatusDescription) {
        this.resultStatusDescription = resultStatusDescription;
    }

    public String getRecommendedCareLocationTimingLocalized() {
        return recommendedCareLocationTimingLocalized;
    }

    public void setRecommendedCareLocationTimingLocalized(String recommendedCareLocationTimingLocalized) {
        this.recommendedCareLocationTimingLocalized = recommendedCareLocationTimingLocalized;
    }

    public String getTriageRatingColorImgURL() {
        return triageRatingColorImgURL;
    }

    public void setTriageRatingColorImgURL(String triageRatingColorImgURL) {
        this.triageRatingColorImgURL = triageRatingColorImgURL;
    }

    public String getTriageRatingCallDoctorTitleLocalized() {
        return triageRatingCallDoctorTitleLocalized;
    }

    public void setTriageRatingCallDoctorTitleLocalized(String triageRatingCallDoctorTitleLocalized) {
        this.triageRatingCallDoctorTitleLocalized = triageRatingCallDoctorTitleLocalized;
    }

    public String getRecommendedCareLocationDesc() {
        return recommendedCareLocationDesc;
    }

    public void setRecommendedCareLocationDesc(String recommendedCareLocationDesc) {
        this.recommendedCareLocationDesc = recommendedCareLocationDesc;
    }

    public String getRecommendedCareLocationDescLocalized() {
        return recommendedCareLocationDescLocalized;
    }

    public void setRecommendedCareLocationDescLocalized(String recommendedCareLocationDescLocalized) {
        this.recommendedCareLocationDescLocalized = recommendedCareLocationDescLocalized;
    }

    public String getRecommendedCareLocationTiming() {
        return recommendedCareLocationTiming;
    }

    public void setRecommendedCareLocationTiming(String recommendedCareLocationTiming) {
        this.recommendedCareLocationTiming = recommendedCareLocationTiming;
    }

    public int getPrimaryCCC() {
        return primaryCCC;
    }

    public void setPrimaryCCC(int primaryCCC) {
        this.primaryCCC = primaryCCC;
    }

    public String getTriageRatingCallDoctorTitle() {
        return triageRatingCallDoctorTitle;
    }

    public void setTriageRatingCallDoctorTitle(String triageRatingCallDoctorTitle) {
        this.triageRatingCallDoctorTitle = triageRatingCallDoctorTitle;
    }

    public String getRecommendedCareLocalized() {
        return recommendedCareLocalized;
    }

    public void setRecommendedCareLocalized(String recommendedCareLocalized) {
        this.recommendedCareLocalized = recommendedCareLocalized;
    }

    public int getRecommendedCareLocationCost4() {
        return recommendedCareLocationCost4;
    }

    public void setRecommendedCareLocationCost4(int recommendedCareLocationCost4) {
        this.recommendedCareLocationCost4 = recommendedCareLocationCost4;
    }

    public String getTriageRatingSeekCareTitle() {
        return triageRatingSeekCareTitle;
    }

    public void setTriageRatingSeekCareTitle(String triageRatingSeekCareTitle) {
        this.triageRatingSeekCareTitle = triageRatingSeekCareTitle;
    }

    public String getTriageRatingSeekCareTitleLocalized() {
        return triageRatingSeekCareTitleLocalized;
    }

    public void setTriageRatingSeekCareTitleLocalized(String triageRatingSeekCareTitleLocalized) {
        this.triageRatingSeekCareTitleLocalized = triageRatingSeekCareTitleLocalized;
    }
}