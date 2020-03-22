package com.healthnavigatorapis.portal.chatbot.data.remote.model;

import com.google.gson.annotations.SerializedName;

public class Question {

    @SerializedName("OPQRSTGroupQuestionPL")
    private String oPQRSTGroupQuestionPL;

    @SerializedName("ResultStatus")
    private int resultStatus;

    @SerializedName("OPQRSTGroupLetter")
    private String oPQRSTGroupLetter;

    @SerializedName("ResultStatusDescription")
    private String resultStatusDescription;

    @SerializedName("OPQRSTGroupQuestionPLLocalized")
    private String oPQRSTGroupQuestionPLLocalized;

    @SerializedName("Title")
    private String title;

    @SerializedName("OPQRSTGroupID")
    private int oPQRSTGroupID;

    @SerializedName("EvaluationAcuityRatingColorImg")
    private String evaluationAcuityRatingColorImg;

    @SerializedName("ConceptId")
    private int conceptId;

    @SerializedName("EvaluationAcuityRatingNumeric")
    private int evaluationAcuityRatingNumeric;

    @SerializedName("ConceptOrder")
    private int conceptOrder;

    @SerializedName("TitlePlainLanguage")
    private String titlePlainLanguage;

    @SerializedName("_inputConceptId")
    private int inputConceptId;

    @SerializedName("FrequencyRatingNumeric")
    private int frequencyRatingNumeric;

    @SerializedName("OPQRSTGroupTitle")
    private String oPQRSTGroupTitle;

    @SerializedName("OPQRSTGroupTitleLocalized")
    private String oPQRSTGroupTitleLocalized;

    @SerializedName("TitleLocalized")
    private String titleLocalized;

    public String getOPQRSTGroupQuestionPL() {
        return oPQRSTGroupQuestionPL;
    }

    public void setOPQRSTGroupQuestionPL(String oPQRSTGroupQuestionPL) {
        this.oPQRSTGroupQuestionPL = oPQRSTGroupQuestionPL;
    }

    public int getResultStatus() {
        return resultStatus;
    }

    public void setResultStatus(int resultStatus) {
        this.resultStatus = resultStatus;
    }

    public String getOPQRSTGroupLetter() {
        return oPQRSTGroupLetter;
    }

    public void setOPQRSTGroupLetter(String oPQRSTGroupLetter) {
        this.oPQRSTGroupLetter = oPQRSTGroupLetter;
    }

    public String getResultStatusDescription() {
        return resultStatusDescription;
    }

    public void setResultStatusDescription(String resultStatusDescription) {
        this.resultStatusDescription = resultStatusDescription;
    }

    public String getOPQRSTGroupQuestionPLLocalized() {
        return oPQRSTGroupQuestionPLLocalized;
    }

    public void setOPQRSTGroupQuestionPLLocalized(String oPQRSTGroupQuestionPLLocalized) {
        this.oPQRSTGroupQuestionPLLocalized = oPQRSTGroupQuestionPLLocalized;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public int getOPQRSTGroupID() {
        return oPQRSTGroupID;
    }

    public void setOPQRSTGroupID(int oPQRSTGroupID) {
        this.oPQRSTGroupID = oPQRSTGroupID;
    }

    public String getEvaluationAcuityRatingColorImg() {
        return evaluationAcuityRatingColorImg;
    }

    public void setEvaluationAcuityRatingColorImg(String evaluationAcuityRatingColorImg) {
        this.evaluationAcuityRatingColorImg = evaluationAcuityRatingColorImg;
    }

    public int getConceptId() {
        return conceptId;
    }

    public void setConceptId(int conceptId) {
        this.conceptId = conceptId;
    }

    public int getEvaluationAcuityRatingNumeric() {
        return evaluationAcuityRatingNumeric;
    }

    public void setEvaluationAcuityRatingNumeric(int evaluationAcuityRatingNumeric) {
        this.evaluationAcuityRatingNumeric = evaluationAcuityRatingNumeric;
    }

    public int getConceptOrder() {
        return conceptOrder;
    }

    public void setConceptOrder(int conceptOrder) {
        this.conceptOrder = conceptOrder;
    }

    public String getTitlePlainLanguage() {
        return titlePlainLanguage;
    }

    public void setTitlePlainLanguage(String titlePlainLanguage) {
        this.titlePlainLanguage = titlePlainLanguage;
    }

    public int getInputConceptId() {
        return inputConceptId;
    }

    public void setInputConceptId(int inputConceptId) {
        this.inputConceptId = inputConceptId;
    }

    public int getFrequencyRatingNumeric() {
        return frequencyRatingNumeric;
    }

    public void setFrequencyRatingNumeric(int frequencyRatingNumeric) {
        this.frequencyRatingNumeric = frequencyRatingNumeric;
    }

    public String getOPQRSTGroupTitle() {
        return oPQRSTGroupTitle;
    }

    public void setOPQRSTGroupTitle(String oPQRSTGroupTitle) {
        this.oPQRSTGroupTitle = oPQRSTGroupTitle;
    }

    public String getOPQRSTGroupTitleLocalized() {
        return oPQRSTGroupTitleLocalized;
    }

    public void setOPQRSTGroupTitleLocalized(String oPQRSTGroupTitleLocalized) {
        this.oPQRSTGroupTitleLocalized = oPQRSTGroupTitleLocalized;
    }

    public String getTitleLocalized() {
        return titleLocalized;
    }

    public void setTitleLocalized(String titleLocalized) {
        this.titleLocalized = titleLocalized;
    }
}